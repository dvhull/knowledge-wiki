# Agent instructions

Do **not** edit the core agent/runtime sources ever. Treat these as read-only defaults:

- `agent.py`
- `environment.py`
- `llm.py`
- `system_prompt.py`
- `tools.py`

You may read them for context. Put new code (for coding agent at runtime not during devlopment.), experiments, docs, in the `scratch/` folder.

Before running code check to see if you have the proper libraries installed. If they are not ask the user before `pip` installing. 

Do not ask for human in the loop except for pip installing something from the internet.