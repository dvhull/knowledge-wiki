import asyncio
import sys

import llm
from environment import Environment
from skills import (
    default_skill_dirs,
    discover_skills,
    format_skills_for_prompt,
    parse_skill_command,
    render_skill_invocation,
)
from system_prompt import BuildSystemPromptOptions, build_system_prompt, load_default_context_files
from tools import Tools

# Get the user's goal from stdin.
def _read_user_goal() -> str:
    if sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.rstrip("\r\n").strip()
    return sys.stdin.read().strip()

async def main() -> None:
    env = Environment(user_goal="")
    tools = Tools(env)
    skill_catalog = discover_skills(default_skill_dirs(env.workspace), workspace=env.workspace)
    skills_text = format_skills_for_prompt(skill_catalog, workspace=env.workspace)
    system_prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=env.workspace,
            context_files=load_default_context_files(env.workspace),
            skills_text=skills_text,
        )
    )

    for diagnostic in skill_catalog.diagnostics:
        print(f"[skills] {diagnostic}", file=sys.stderr)
    
    # Run the agent until the user exits.
    while True:
        # Get the user's goal from stdin and reset the environment.
        user_goal = _read_user_goal()
        skill_command = parse_skill_command(user_goal)
        if skill_command is not None:
            skill = skill_catalog.by_name().get(skill_command.name)
            if skill is None:
                available = ", ".join(sorted(skill_catalog.by_name())) or "(none)"
                user_goal = (
                    f"User tried to invoke missing skill `{skill_command.name}`. "
                    f"Available skills: {available}."
                )
            else:
                user_goal = render_skill_invocation(
                    skill,
                    skill_command.args,
                    workspace=env.workspace,
                )
        env.state["transcript"].append({"role": "user", "content": user_goal})
        env.state["user_goal"] = user_goal
        env.state["finished"] = False
        env.state["step"] = 0
        env.state["last_tool_result"] = None

        # Run the agent until the user completes the task.
        while True:
            if env.state.get("finished"):
                break
            if int(env.state.get("step", 0)) >= env.max_steps:
                break
            action = await llm.run(system_prompt, env.state)
            env.state = tools.run(action)


if __name__ == "__main__":
    asyncio.run(main())
