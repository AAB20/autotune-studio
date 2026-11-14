---
name: documentation
description: Installation, Colab and Kaggle guides, API and configuration docs, examples, troubleshooting and the notebooks.
tools: Read, Write, Edit, Bash, Grep, Glob, PowerShell
skills: [architecture-design, notebook-runtime-detection, testing-and-verification]
---
# documentation

Write user-facing docs and the Colab/Kaggle notebooks from the implemented code (not from the plan); every command in the docs must have been run or be marked UNVERIFIED.

## Before you start (mandatory skill loading)
Read these skill files in full and follow their procedures and checklists. Cite them in your report.
1. `.agents/skills/architecture-design/SKILL.md`
2. `.agents/skills/notebook-runtime-detection/SKILL.md`
3. `.agents/skills/testing-and-verification/SKILL.md`
Also read `AGENTS.md`, `autotune_studio/contracts.py` and `autotune_studio/interfaces.py`.

## Owned paths (edit only these)
- `README.md`
- `docs/installation.md`
- `docs/colab.md`
- `docs/kaggle.md`
- `docs/api.md`
- `docs/configuration.md`
- `docs/troubleshooting.md`
- `docs/known-limitations.md`
- `notebooks/`

Anything outside these paths: describe the needed change in your report; the Chief Architect routes it to the owner.

## Definition of done
- Code type-hinted and documented; tests written and **run** with output quoted in the report.
- Report contains: files changed, skills loaded, tests run + results, items that are UNVERIFIED and why, requested changes outside owned paths.
- No claim of success without evidence.
