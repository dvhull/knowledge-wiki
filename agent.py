import asyncio
import sys

import llm
from environment import Environment
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
    system_prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=env.workspace,
            context_files=load_default_context_files(env.workspace),
        )
    )
    
    # Run the agent until the user exits.
    while True:
        # Get the user's goal from stdin and reset the environment.
        user_goal = _read_user_goal()
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
