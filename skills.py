"""Pi-style skill discovery and slash-command invocation.

Skills are Markdown instruction packs stored under `.agents/skills`. The agent
keeps only stable skill metadata in the system prompt and loads full SKILL.md
content on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import textwrap

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    file_path: Path
    base_dir: Path
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[Skill, ...]
    diagnostics: tuple[str, ...] = ()

    def by_name(self) -> dict[str, Skill]:
        return {skill.name: skill for skill in self.skills}


@dataclass(frozen=True)
class SkillCommand:
    name: str
    args: str


@dataclass(frozen=True)
class SkillScript:
    skill_name: str
    relative_path: str
    file_path: Path


@dataclass(frozen=True)
class ScriptCommand:
    action: str
    query: str = ""
    skill_name: str = ""
    relative_path: str = ""
    args: tuple[str, ...] = ()


def default_skill_dirs(workspace: Path) -> tuple[Path, ...]:
    """Return project-local skill roots."""

    return (workspace / ".agents" / "skills",)


def discover_skills(skill_dirs: tuple[Path, ...] | list[Path], *, workspace: Path) -> SkillCatalog:
    """Discover all `SKILL.md` files beneath the given skill directories."""

    workspace = workspace.resolve()
    diagnostics: list[str] = []
    discovered: list[Skill] = []

    for skill_dir in skill_dirs:
        skill_dir = skill_dir.resolve()
        if not skill_dir.exists():
            continue
        if not _is_within(skill_dir, workspace):
            diagnostics.append(f"ignored skill directory outside workspace: {skill_dir}")
            continue
        for skill_file in _find_skill_files(skill_dir):
            skill = _load_skill(skill_file, diagnostics)
            if skill is not None:
                discovered.append(skill)

    by_name: dict[str, Skill] = {}
    for skill in sorted(discovered, key=lambda s: (_display_path(s.file_path, workspace), s.name)):
        if skill.name in by_name:
            diagnostics.append(
                "duplicate skill name "
                f"{skill.name!r}; keeping {_display_path(by_name[skill.name].file_path, workspace)} "
                f"and ignoring {_display_path(skill.file_path, workspace)}"
            )
            continue
        by_name[skill.name] = skill

    return SkillCatalog(skills=tuple(by_name.values()), diagnostics=tuple(diagnostics))


def format_skills_for_prompt(catalog: SkillCatalog, *, workspace: Path) -> str:
    """Format a compact, KV-cache-friendly skills section."""

    visible = [skill for skill in catalog.skills if not skill.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "",
        "",
        "# Skills",
        "",
        "Skills are optional instruction packs. Use them when their description matches the user's task.",
        "Do not preload full skill files. When a skill is relevant, use the read tool on its SKILL.md path, then follow its instructions.",
        "",
        "Available skills:",
    ]
    for skill in visible:
        path = _display_path(skill.file_path, workspace)
        lines.append(f"- {skill.name}: {skill.description} (file: {path})")
    return "\n".join(lines)


def parse_skill_command(text: str) -> SkillCommand | None:
    """Parse `/skill:name args` or `/skill-name args`."""

    stripped = text.strip()
    if not stripped.startswith("/skill"):
        return None

    rest = stripped[len("/skill") :]
    if rest.startswith(":") or rest.startswith("-"):
        rest = rest[1:]
    else:
        return None

    name, _, args = rest.partition(" ")
    name = name.strip()
    if not name or not _VALID_NAME.match(name):
        return None
    return SkillCommand(name=name, args=args.strip())


def parse_script_command(text: str) -> ScriptCommand | None:
    """Parse direct Python script commands for discovered skills.

    Supported forms:
    - `/scripts [query]`
    - `/script list [query]`
    - `/script:list [query]`
    - `/script <skill>/<relative/path.py> [args...]`
    - `/script:<skill>/<relative/path.py> [args...]`
    - `/script <skill> [args...]` when the skill has exactly one Python script
    - `/script:<skill> [args...]` when the skill has exactly one Python script
    """

    stripped = text.strip()
    if stripped == "/scripts" or stripped.startswith("/scripts "):
        _, _, query = stripped.partition(" ")
        return ScriptCommand(action="list", query=query.strip())

    if not stripped.startswith("/script"):
        return None

    rest = stripped[len("/script") :]
    if rest.startswith(":"):
        rest = rest[1:].strip()
    elif rest.startswith(" "):
        rest = rest.strip()
    else:
        return None
    if not rest:
        return ScriptCommand(action="list")

    head, _, tail = rest.partition(" ")
    if head == "list":
        return ScriptCommand(action="list", query=tail.strip())

    try:
        parts = shlex.split(rest)
    except ValueError:
        return None
    if not parts:
        return ScriptCommand(action="list")

    target = parts[0]
    args = tuple(parts[1:])

    if "/" in target:
        skill_name, relative_path = target.split("/", 1)
    else:
        skill_name, relative_path = target, ""

    skill_name = skill_name.strip()
    relative_path = relative_path.strip()
    if not skill_name or not _VALID_NAME.match(skill_name):
        return None
    if relative_path and not relative_path.endswith(".py"):
        return None
    return ScriptCommand(
        action="run",
        skill_name=skill_name,
        relative_path=relative_path,
        args=args,
    )


def render_skill_invocation(skill: Skill, args: str, *, workspace: Path) -> str:
    """Load full skill content for an explicit slash invocation."""

    content = skill.file_path.read_text(encoding="utf-8", errors="replace").rstrip()
    path = _display_path(skill.file_path, workspace)
    rendered = f"""The user explicitly invoked skill `{skill.name}`.

<skill name="{skill.name}" path="{path}">
{content}
</skill>

User request for this skill:
{args.strip() or "(no additional request text provided)"}
"""
    return textwrap.dedent(rendered).strip()


def discover_python_scripts(catalog: SkillCatalog, query: str = "") -> tuple[SkillScript, ...]:
    """Find Python scripts contained inside discovered skill directories."""

    normalized_query = query.strip().lower()
    scripts: list[SkillScript] = []
    for skill in catalog.skills:
        for path in sorted(skill.base_dir.rglob("*.py")):
            try:
                relative = path.resolve().relative_to(skill.base_dir).as_posix()
            except ValueError:
                continue
            if any(part in _IGNORED_DIRS or part.startswith(".") for part in Path(relative).parts):
                continue
            script = SkillScript(
                skill_name=skill.name,
                relative_path=relative,
                file_path=path.resolve(),
            )
            haystack = f"{script.skill_name} {script.relative_path}".lower()
            if normalized_query and normalized_query not in haystack:
                continue
            scripts.append(script)

    return tuple(sorted(scripts, key=lambda s: (s.skill_name, s.relative_path)))


def resolve_python_script(catalog: SkillCatalog, command: ScriptCommand) -> SkillScript:
    """Resolve a run command to a Python script within a skill directory."""

    if command.action != "run":
        raise ValueError("script command is not a run command")
    skill = catalog.by_name().get(command.skill_name)
    if skill is None:
        raise ValueError(f"unknown skill: {command.skill_name}")

    scripts = [script for script in discover_python_scripts(catalog) if script.skill_name == skill.name]
    if command.relative_path:
        for script in scripts:
            if script.relative_path == command.relative_path:
                return script
        raise ValueError(f"script not found in skill {skill.name}: {command.relative_path}")

    if len(scripts) == 1:
        return scripts[0]
    if not scripts:
        raise ValueError(f"skill {skill.name} has no Python scripts")
    raise ValueError(
        f"skill {skill.name} has multiple Python scripts; specify one like "
        f"/script:{skill.name}/{scripts[0].relative_path}"
    )


def _find_skill_files(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]

    while stack:
        current = stack.pop()
        if current.name in _IGNORED_DIRS:
            continue
        skill_file = current / "SKILL.md"
        if skill_file.is_file():
            out.append(skill_file)
            continue
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        stack.extend(reversed([child for child in children if child.is_dir()]))

    return sorted(out)


def _load_skill(skill_file: Path, diagnostics: list[str]) -> Skill | None:
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        diagnostics.append(f"could not read {skill_file}: {exc}")
        return None

    metadata = _parse_frontmatter(content)
    name = metadata.get("name", skill_file.parent.name).strip()
    description = metadata.get("description", "").strip()
    disable = metadata.get("disable-model-invocation") or metadata.get(
        "disable_model_invocation", "false"
    )

    if not name:
        diagnostics.append(f"ignored {skill_file}: missing skill name")
        return None
    if not _VALID_NAME.match(name):
        diagnostics.append(f"ignored {skill_file}: invalid skill name {name!r}")
        return None
    if not description:
        diagnostics.append(f"ignored {skill_file}: missing description")
        return None

    return Skill(
        name=name,
        description=description,
        file_path=skill_file.resolve(),
        base_dir=skill_file.parent.resolve(),
        disable_model_invocation=_as_bool(disable),
    )


def _parse_frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---", 4)
    if end == -1:
        return {}

    metadata: dict[str, str] = {}
    for line in content[4:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
