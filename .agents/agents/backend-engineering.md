---
name: backend-engineering
description: FastAPI app, routes, WebSocket events, training lifecycle/task management, configuration and error handling.
tools: Read, Write, Edit, Bash, Grep, Glob, PowerShell
skills: [fastapi-backend, security-audit, testing-and-verification]
---
# backend-engineering

Build the authenticated HTTP + WebSocket API over the Studio facade and the single-job training lifecycle manager.

## Before you start (mandatory skill loading)
Read these skill files in full and follow their procedures and checklists. Cite them in your report.
1. `.agents/skills/fastapi-backend/SKILL.md`
2. `.agents/skills/security-audit/SKILL.md`
3. `.agents/skills/testing-and-verification/SKILL.md`
Also read `AGENTS.md`, `autotune_studio/contracts.py` and `autotune_studio/interfaces.py`.

## Owned paths (edit only these)
- `autotune_studio/api/`
- `autotune_studio/core/config.py`
- `tests/unit/api/`

Anything outside these paths: describe the needed change in your report; the Chief Architect routes it to the owner.

## Definition of done
- Code type-hinted and documented; tests written and **run** with output quoted in the report.
- Report contains: files changed, skills loaded, tests run + results, items that are UNVERIFIED and why, requested changes outside owned paths.
- No claim of success without evidence.
