---
name: example
description: Demonstrates explicit skill invocation for this minimalist coding agent.
disable-model-invocation: true
---

# Example Skill

Use this skill only when explicitly invoked with `/skill:example`.

When invoked, briefly confirm that the skill was loaded and summarize the user
request that followed the slash command.

This skill also includes a separate Python script stored at
`scripts/hello.py`. The script is not embedded in this Markdown file and does
not run when the skill is loaded.

```bash
python .agents/skills/example/scripts/hello.py hello
```
