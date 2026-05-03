"""
System prompt construction: tools list, guidelines, optional project files, date/cwd.

Date and cwd are appended at the *end* so the leading instructions stay stable within a day
for prefix / prompt caching; avoid putting timestamps at the very top of the prompt.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

# One-line snippets shown in the default system prompt (aligned with `llm.TOOLS`).
_DEFAULT_TOOL_SNIPPETS: dict[str, str] = {
    "read": (
        "Read UTF-8 text (optional 1-based offset/limit) or common images as base64; "
        "paths relative to workspace."
    ),
    "write": "Create or overwrite a UTF-8 file; creates parent directories.",
    "edit": (
        "Surgical edits: path + edits[{oldText,newText}, ...] against the original file; "
        "each oldText unique and non-overlapping."
    ),
    "bash": "Run shell via bash -lc in workspace; optional timeout (seconds); prefer non-interactive commands.",
}

_DEFAULT_TOOL_ORDER = ("read", "write", "edit", "bash")

# Optional project instruction files (first match wins order for loading).
_DEFAULT_CONTEXT_FILENAMES = ("AGENTS.md", "README.md", "SYSTEM.md")


@dataclass
class BuildSystemPromptOptions:
    """Options for `build_system_prompt` (cwd, overrides, context files, skills text)."""

    cwd: Path
    custom_prompt: str | None = None
    selected_tools: tuple[str, ...] | list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] = field(default_factory=list)
    append_system_prompt: str | None = None
    context_files: list[tuple[str, str]] | None = None
    skills_text: str | None = None


def load_default_context_files(cwd: Path, *, max_bytes: int = 80_000) -> list[tuple[str, str]]:
    """Load known instruction files from the workspace if they exist."""
    out: list[tuple[str, str]] = []
    for name in _DEFAULT_CONTEXT_FILENAMES:
        path = cwd / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n\n[File truncated for prompt size.]\n"
        out.append((name, text))
    return out


def _visible_tools(
    tools: tuple[str, ...],
    snippets: dict[str, str],
) -> list[tuple[str, str]]:
    return [(n, snippets[n]) for n in tools if n in snippets]


def _merge_snippets(selected: tuple[str, ...] | None) -> dict[str, str]:
    base = dict(_DEFAULT_TOOL_SNIPPETS)
    tools = tuple(selected) if selected else _DEFAULT_TOOL_ORDER
    for name in tools:
        if name not in base:
            base[name] = f"Tool `{name}` (see tool definitions in the API)."
    return {n: base[n] for n in tools if n in base}


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    resolved = options.cwd.resolve()
    prompt_cwd = str(resolved).replace("\\", "/")
    today = datetime.date.today()
    date = today.isoformat()
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""

    tools: tuple[str, ...] = (
        tuple(options.selected_tools)
        if options.selected_tools
        else tuple(_DEFAULT_TOOL_ORDER)
    )
    merged = _merge_snippets(tools)
    if options.tool_snippets:
        merged.update(options.tool_snippets)
    visible = _visible_tools(tools, merged)
    tools_list = "\n".join(f"- {n}: {s}" for n, s in visible) if visible else "(none)"

    context_files = list(options.context_files) if options.context_files is not None else []

    if options.custom_prompt:
        prompt = options.custom_prompt
        if append_section:
            prompt += append_section
        if context_files:
            prompt += "\n\n# Project Context\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for file_path, content in context_files:
                prompt += f"## {file_path}\n\n{content}\n\n"
        if options.skills_text and "read" in tools:
            prompt += options.skills_text
        prompt += f"\nCurrent date: {date}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    has_bash = "bash" in tools
    has_read = "read" in tools

    guidelines_set: set[str] = set()
    guidelines_list: list[str] = []

    def add_guideline(g: str) -> None:
        g = g.strip()
        if not g or g in guidelines_set:
            return
        guidelines_set.add(g)
        guidelines_list.append(g)

    if has_bash:
        add_guideline("Use bash for file operations like ls, rg, find, git, and package commands.")
    for g in options.prompt_guidelines:
        add_guideline(g)
    add_guideline("Be concise in your responses.")
    add_guideline("Show file paths clearly when working with files.")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = f"""You are an expert coding assistant using a minimal terminal coding agent with four tools: read, write, edit, and bash. You help users by reading files, running commands, editing code, and writing new files.

Available tools:
{tools_list}

Guidelines:
{guidelines}

Prefer project context below when it is more specific than these defaults.
{append_section}"""

    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for file_path, content in context_files:
            prompt += f"## {file_path}\n\n{content}\n\n"

    if has_read and options.skills_text:
        prompt += options.skills_text

    prompt += f"\nCurrent date: {date}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt
