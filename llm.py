"""One OpenAI Responses turn: four core tools + Manus-style context for prefix / prompt-cache friendliness.

KV-cache / prompt-cache practices follow
https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
(stable prefixes, deterministic serialization, append-only observations; fixed tool list).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Identical on every `LLM.run` call so provider prefix caching can reuse work from turn to turn.
# Keep this free of timestamps, session ids, or other per-request drift (Manus: stable prefix).
_STATIC_USER_PROTOCOL = (
    "Agent protocol (fixed).\n"
    "You may call tools: read, write, edit, bash.\n"
    "Use read for file contents; use edit with path + edits[{oldText,newText}] for precise edits.\n"
    "When finished, respond with normal assistant text and no tool calls.\n"
    "All file paths are relative to workspace_root in the state JSON.\n"
)

_STATE_BANNER = "Current agent state (JSON, append-only transcript; do not rewrite prior steps):\n"

# Four core filesystem / shell tools; JSON schemas match `tools.py` dispatch.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "read",
        "description": (
            "Read the contents of a file. Text is truncated to 2000 lines or 50KB (whichever first); "
            "use offset (1-based line) and limit (max lines) to page through large files. "
            "Common image types are returned as base64 in a text note (no separate image channel)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace"},
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from",
                },
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write",
        "description": (
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "edit",
        "description": (
            "Edit a file using exact text replacements. Each edits[].oldText must match a unique, "
            "non-overlapping region of the original file (matches are against the file as it was "
            "before this call, not incrementally). Merge nearby changes into one edit when needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                    },
                    "description": "One or more replacements applied to the original file content",
                },
            },
            "required": ["path", "edits"],
        },
    },
    {
        "type": "function",
        "name": "bash",
        "description": (
            "Execute a bash command in the workspace root. Returns stdout and stderr; "
            "long output is tail-truncated (default 2000 lines / 50KB caps). "
            "Optionally set timeout in seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command passed to bash -lc"},
                "timeout": {"type": "number", "description": "Optional timeout in seconds"},
            },
            "required": ["command"],
        },
    },
]


def _serialize_state(state: Mapping[str, Any], *, max_chars: int = 60_000) -> str:
    """Deterministic JSON for stable prefixes across turns (Manus: deterministic serialization)."""
    text = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[state json truncated]"


class LLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        self.client = AsyncOpenAI()

    async def run(self, system_prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the model once; it may return native `function_call` items from `TOOLS`."""
        instructions = system_prompt.strip() or None
        payload = _serialize_state(state)
        # Two user messages: (1) byte-stable prefix for KV / prompt cache, (2) growing state only.
        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "user",
                "content": _STATIC_USER_PROTOCOL,
            },
            {
                "type": "message",
                "role": "user",
                "content": _STATE_BANNER + payload,
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": TOOLS,
            "temperature": 0.0,
        }
        if instructions:
            kwargs["instructions"] = instructions

        response = await self.client.responses.create(**kwargs)
        st = response.status
        if st is not None and st != "completed":
            err = response.error
            detail = getattr(err, "message", str(err)) if err else "(no error object)"
            raise RuntimeError(f"OpenAI response status={st}: {detail}")

        tool_calls: list[dict[str, Any]] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            raw = getattr(item, "arguments", "") or ""
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append({"name": item.name, "arguments": args})

        assistant_message = (response.output_text or "").strip()
        logger.debug("llm.run: %d tool call(s), text_len=%d", len(tool_calls), len(assistant_message))

        return {
            "tool_calls": tool_calls,
            "assistant_message": assistant_message,
        }


_default: LLM | None = None


def _get() -> LLM:
    global _default
    if _default is None:
        _default = LLM()
    return _default


async def run(system_prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
    return await _get().run(system_prompt, state)
