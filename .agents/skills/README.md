# Skills

Place Pi-style skills in this folder.

Each skill is a directory with a `SKILL.md` file:

```text
.agents/
  skills/
    example/
      SKILL.md
      scripts/
      references/
      assets/
```

`SKILL.md` should start with frontmatter:

```markdown
---
name: example
description: Use this skill when ...
---
```

The agent keeps only skill names, descriptions, and `SKILL.md` paths in the
system prompt. Full skill content is loaded later with the `read` tool, or by
explicitly invoking `/skill:name your request`.

Put skill-specific helper scripts inside that skill's folder so the skill stays
portable.

The included `example` skill has `disable-model-invocation: true`, so it is
available for `/skill:example ...` testing without appearing in the automatic
skill list.
