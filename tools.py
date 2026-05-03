"""Run tool calls returned by the LLM (names must match `llm.TOOLS`)."""

from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from environment import Environment

# Default read/bash output limits (2000 lines, 50KB) — common agent defaults.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
_READ_MAX_B64_CHARS = 400_000
_MAX_TERMINAL_TOOL_OUTPUT = 12_000

def _log_assistant_to_terminal(content: str) -> None:
    """Echo assistant text to stderr (same channel as tool logs)."""
    n = len(content)
    show = content
    if len(show) > _MAX_TERMINAL_TOOL_OUTPUT:
        show = (
            show[:_MAX_TERMINAL_TOOL_OUTPUT]
            + f"\n…[truncated for terminal, {n} chars total]"
        )
    print(
        f"\n{'=' * 60}\nASSISTANT\n{'=' * 60}\n{show}\n",
        file=sys.stderr,
        flush=True,
    )


def _log_tool_to_terminal(name: str, args: dict, out: str) -> None:
    """Echo tool name, arguments, and output to stderr (keeps `agent.py` free of print logic)."""
    try:
        args_text = json.dumps(args, default=str, ensure_ascii=False, indent=2)
    except TypeError:
        args_text = repr(args)
    if len(args_text) > 6_000:
        args_text = args_text[:6_000] + "\n…[arguments truncated]"

    out_len = len(out)
    out_show = out
    if len(out_show) > _MAX_TERMINAL_TOOL_OUTPUT:
        out_show = (
            out_show[:_MAX_TERMINAL_TOOL_OUTPUT]
            + f"\n…[output truncated for terminal, {out_len} chars total]"
        )

    print(
        f"\n{'=' * 60}\nTOOL {name}\n{'=' * 60}\narguments:\n{args_text}\n"
        f"--- output ({out_len} chars) ---\n{out_show}\n",
        file=sys.stderr,
        flush=True,
    )


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple:
    """Returns (output_text, meta dict with truncated, first_line_exceeds, etc.)."""
    lines = content.split("\n")
    total_lines = len(lines)
    total_bytes = len(content.encode("utf-8"))
    meta: dict = {"truncated": False, "truncated_by": None, "total_lines": total_lines}

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content, meta

    first = lines[0] if lines else ""
    if len(first.encode("utf-8")) > max_bytes:
        meta["truncated"] = True
        meta["truncated_by"] = "bytes"
        meta["first_line_exceeds"] = True
        return "", meta

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            truncated_by = "lines"
            break
        extra = 1 if i > 0 else 0
        b = len(line.encode("utf-8")) + extra
        if out_bytes + b > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += b

    meta["truncated"] = True
    meta["truncated_by"] = truncated_by
    return "\n".join(out_lines), meta


def _truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, dict]:
    """Keep the end of output within line/byte caps (tail truncation for bash)."""
    meta: dict = {"truncated": False, "total_lines": 0, "output_lines": 0}
    if not content:
        return "", meta
    lines = content.split("\n")
    meta["total_lines"] = len(lines)
    total_bytes = len(content.encode("utf-8"))
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return content, meta
    chunk = lines[-max_lines:]
    out = "\n".join(chunk)
    while chunk and len(out.encode("utf-8")) > max_bytes:
        chunk = chunk[1:]
        out = "\n".join(chunk)
    meta["truncated"] = True
    meta["output_lines"] = len(chunk)
    return out, meta


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s, 10)
        except ValueError:
            return None
    return None


def _is_probably_image(path: Path, data: bytes) -> bool:
    suf = path.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return True
    return data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8\xff") or data.startswith(b"GIF8")


class Tools:
    def __init__(self, env: Environment) -> None:
        self.env = env

    def _safe_path(self, rel: str) -> Path:
        root = self.env.workspace
        candidate = rel.strip() or "."
        target = (root / candidate).resolve()
        target.relative_to(root)
        return target

    def read(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read file: optional 1-based `offset` line, optional `limit` lines; then head truncation."""
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        data = target.read_bytes()
        if _is_probably_image(target, data):
            mime = "image/png"
            if target.suffix.lower() in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif target.suffix.lower() == ".gif":
                mime = "image/gif"
            elif target.suffix.lower() == ".webp":
                mime = "image/webp"
            b64 = base64.standard_b64encode(data).decode("ascii")
            if len(b64) > _READ_MAX_B64_CHARS:
                b64 = b64[:_READ_MAX_B64_CHARS] + "\n…[base64 truncated]"
            note = f"Read image file [{mime}]"
            if offset is not None or limit is not None:
                note += " (offset/limit ignored for images)"
            return f"{note}\n{b64}"

        if b"\x00" in data[:8192]:
            return f"[binary non-image file, {len(data)} bytes; not shown as text]"
        try:
            full_text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return f"[file is not valid UTF-8 text, {len(data)} bytes]"

        all_lines = full_text.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (offset or 1) - 1)
        if start_line >= len(all_lines):
            raise ValueError(
                f"Offset {offset} is beyond end of file ({total_file_lines} lines total)"
            )
        start_display = start_line + 1

        if limit is not None:
            chunk_lines = all_lines[start_line : start_line + int(limit)]
            selected = "\n".join(chunk_lines)
            user_limited = True
        else:
            selected = "\n".join(all_lines[start_line:])
            user_limited = False

        body, tmeta = _truncate_head(selected)
        parts: list[str] = []

        if tmeta.get("first_line_exceeds"):
            parts.append(
                f"[Line {start_display} is {_format_size(len(all_lines[start_line].encode('utf-8')))}, "
                f"exceeds {_format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            return "\n".join(parts)

        parts.append(body)

        if tmeta["truncated"]:
            out_lines = body.count("\n") + (1 if body else 0)
            end_display = start_display + out_lines - 1
            next_offset = end_display + 1
            tb = tmeta.get("truncated_by")
            if tb == "lines":
                parts.append(
                    f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                parts.append(
                    f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines} "
                    f"({_format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
        elif user_limited and limit is not None and start_line + int(limit) < len(all_lines):
            remaining = len(all_lines) - (start_line + int(limit))
            next_offset = start_line + int(limit) + 1
            parts.append(
                f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        return "".join(parts)

    def write(self, path: str, content: str) -> str:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        nbytes = len(content.encode("utf-8"))
        return f"Successfully wrote {nbytes} bytes to {path}"

    def edit(self, path: str, edits: list[dict]) -> str:
        """Edit file: `edits` apply to the original content; each oldText unique; spans must not overlap."""
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        raw = target.read_text(encoding="utf-8")
        pairs: list[tuple[str, str]] = []
        for e in edits:
            if not isinstance(e, dict):
                raise ValueError("each edit must be an object")
            old = e.get("oldText")
            new = e.get("newText", "")
            if not isinstance(old, str) or old == "":
                raise ValueError("each edit needs non-empty oldText")
            if not isinstance(new, str):
                new = str(new)
            pairs.append((old, new))

        spans: list[tuple[int, int, str]] = []
        for old, new in pairs:
            c = raw.count(old)
            if c == 0:
                raise ValueError("oldText not found in file")
            if c > 1:
                raise ValueError(
                    f"oldText appears {c} times; must be unique in the original file"
                )
            idx = raw.index(old)
            spans.append((idx, idx + len(old), new))

        spans.sort(key=lambda s: s[0])
        for i in range(len(spans) - 1):
            if spans[i][1] > spans[i + 1][0]:
                raise ValueError("overlapping edits (merge into one edit or split steps)")

        text = raw
        for start, end, new in sorted(spans, key=lambda s: s[0], reverse=True):
            if text[start:end] != raw[start:end]:
                raise ValueError("edit span does not match original file")
            text = text[:start] + new + text[end:]

        target.write_text(text, encoding="utf-8")
        return f"Successfully replaced {len(pairs)} block(s) in {path}."

    def bash(self, command: str, timeout: float | None = None) -> str:
        cmd = str(command).strip()
        if not cmd:
            raise ValueError("empty bash command")
        run_args: list[str] = ["/bin/bash", "-lc", cmd]
        kwargs: dict = {
            "cwd": str(self.env.workspace),
            "capture_output": True,
            "text": True,
            "env": os.environ.copy(),
        }
        if timeout is not None and timeout > 0:
            kwargs["timeout"] = float(timeout)
        try:
            proc = subprocess.run(run_args, **kwargs)
        except subprocess.TimeoutExpired as exc:
            out = ""
            if exc.stdout:
                out += str(exc.stdout)
            if exc.stderr:
                out += "\n--- stderr ---\n" + str(exc.stderr)
            return (out + f"\n\nCommand timed out after {timeout} seconds").strip()

        parts: list[str] = []
        merged = ""
        if proc.stdout:
            merged += proc.stdout
        if proc.stderr:
            if merged:
                merged += "\n"
            merged += proc.stderr
        tail_out, tmeta = _truncate_tail(merged)
        parts.append(tail_out.rstrip("\n") if tail_out else "(no output)")
        if tmeta.get("truncated"):
            tl = tmeta.get("total_lines", 0)
            ol = tmeta.get("output_lines", 0)
            parts.append(
                f"\n\n[Truncated: showing last ~{ol} of {tl} lines "
                f"({_format_size(DEFAULT_MAX_BYTES)} / {DEFAULT_MAX_LINES} line limits).]"
            )
        parts.append(f"\n--- exit code: {proc.returncode} ---")
        out = "\n".join(parts)
        if len(out) > 200_000:
            return out[:200_000] + "\n…[output truncated]"
        return out

    def _normalize_edits_arg(self, args: dict) -> list[dict]:
        raw_edits = args.get("edits")
        if isinstance(raw_edits, str):
            try:
                parsed = json.loads(raw_edits)
                if isinstance(parsed, list):
                    args = {**args, "edits": parsed}
            except json.JSONDecodeError:
                pass
        if isinstance(args.get("edits"), list) and len(args["edits"]) > 0:
            return list(args["edits"])
        o = args.get("oldText") if args.get("oldText") is not None else args.get("old_string")
        n = args.get("newText") if args.get("newText") is not None else args.get("new_string")
        if isinstance(o, str) and o:
            return [{"oldText": o, "newText": n if isinstance(n, str) else ""}]
        raise ValueError("edit requires non-empty edits[] (or legacy oldText/newText)")

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "read":
            p = str(args.get("path") or args.get("file_path") or "").strip()
            if not p:
                raise ValueError("read requires a non-empty path")
            oi = _coerce_optional_int(args.get("offset"))
            li = _coerce_optional_int(args.get("limit"))
            return self.read(p, offset=oi, limit=li)
        if name == "write":
            p = str(args.get("path") or args.get("file_path") or "").strip()
            if not p:
                raise ValueError("write requires a non-empty path")
            return self.write(p, str(args.get("content", "")))
        if name == "edit":
            p = str(args.get("path") or args.get("file_path") or "").strip()
            if not p:
                raise ValueError("edit requires a non-empty path")
            edits = self._normalize_edits_arg(args)
            return self.edit(p, edits)
        if name == "bash":
            to = args.get("timeout")
            tf: float | None = None
            if to is not None:
                try:
                    tf = float(to)
                except (TypeError, ValueError):
                    tf = None
            return self.bash(str(args.get("command", "")), timeout=tf)
        raise ValueError(f"unknown tool: {name}")

    def run(self, step: dict) -> dict:
        """`step` is `{"tool_calls": [...], "assistant_message": str}` from `llm.run`."""
        state = copy.deepcopy(self.env.state)
        state["step"] = int(state.get("step", 0)) + 1

        msg = (step.get("assistant_message") or "").strip()
        if msg:
            _log_assistant_to_terminal(msg)
            state["transcript"].append({"role": "assistant", "content": msg})

        calls = list(step.get("tool_calls") or [])
        if not calls:
            state["finished"] = True
            return state

        for call in calls:
            name = (call.get("name") or "").strip()
            args = dict(call.get("arguments") or {})

            try:
                out = self._dispatch(name, args)
            except Exception as exc:
                out = f"error: {type(exc).__name__}: {exc}"

            _log_tool_to_terminal(name, args, out)
            state["last_tool_result"] = out[:50_000]
            state["transcript"].append({"role": "assistant", "tool": name, "arguments": args})
            state["transcript"].append({"role": "tool", "name": name, "output": out[:20_000]})

        return state
