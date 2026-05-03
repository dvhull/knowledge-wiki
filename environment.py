"""Agent-visible world: workspace cwd and serializable `state`."""

from pathlib import Path

class Environment:
    """Holds `state` the LLM reads each turn (goal, transcript, flags).

    Default workspace is ``Path.cwd()``: start the agent from the directory you want to edit.
    Pass ``workspace=`` only for tests or embedding.
    """

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        user_goal: str = "",
        max_steps: int = 64,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.max_steps = max_steps
        self.state: dict = {
            "user_goal": user_goal,
            "workspace_root": str(self.workspace),
            "step": 0,
            "max_steps": max_steps,
            "transcript": [],
            "last_tool_result": None,
            "finished": False,
        }
