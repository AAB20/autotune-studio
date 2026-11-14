# AGENTS.md — AutoTune Studio multi-agent development

AutoTune Studio is built by a hierarchical team of specialised agents coordinated by the **Chief Architect**.
Agent definitions: `.agents/agents/*.md`. Skills: `.agents/skills/*/SKILL.md`.

## Skill discovery and loading
Creating a SKILL.md does not activate it. Loading is explicit:

1. `python scripts/sync_agents.py` validates every agent and skill (required sections, skill references) and mirrors them into `.claude/agents` / `.claude/skills`, where Claude Code discovers them natively.
2. Every agent definition lists its skills in frontmatter (`skills: [...]`) and its body requires reading those SKILL.md files before work. The coordinator's dispatch prompt repeats the list and the agent's report must cite the skills it loaded.
3. `tests/unit/test_agent_config.py` fails CI if an agent references a missing skill or a skill lacks a required section.

## Roster and ownership (edit only your paths)

| Agent | Owns | Skills |
|---|---|---|
| chief-architect | `AGENTS.md`, `pyproject.toml`, `requirements/`, `autotune_studio/{__init__,contracts,interfaces,errors,studio,cli}.py`, `autotune_studio/core/{events,lifecycle}.py`, `tests/unit/core/`, `docs/architecture.md`, `docs/agent-workflow.md`, `docs/decisions/`, `scripts/` | architecture-design, testing-and-verification, security-audit |
| data-engineering | `autotune_studio/connectors/`, `autotune_studio/data_engine/`, `tests/unit/data/` | streaming-data-engineering, dataset-validation, testing-and-verification |
| runtime-compatibility | `autotune_studio/core/{environment,hardware}.py`, `autotune_studio/launch/`, `tests/unit/runtime/` | notebook-runtime-detection, security-audit, testing-and-verification |
| backend-engineering | `autotune_studio/api/`, `autotune_studio/core/config.py`, `tests/unit/api/` | fastapi-backend, security-audit, testing-and-verification |
| frontend-engineering | `frontend/` | react-dashboard, testing-and-verification |
| llm-training | `autotune_studio/training/`, `autotune_studio/export/`, `tests/unit/training/` | llm-finetuning, gpu-memory-optimization, testing-and-verification |
| gpu-memory-optimization | `autotune_studio/monitoring/`, `tests/unit/monitoring/` | gpu-memory-optimization, testing-and-verification |
| qa-testing | `tests/integration/`, `tests/compatibility/`, `tests/conftest.py`, `docs/acceptance.md`, `docs/progress-report.md` | testing-and-verification, streaming-data-engineering, fastapi-backend |
| security | `autotune_studio/security/`, `tests/unit/security/`, `docs/security-review.md` | security-audit, testing-and-verification |
| documentation | `README.md`, `docs/{installation,colab,kaggle,api,configuration,troubleshooting,known-limitations}.md`, `notebooks/` | architecture-design, notebook-runtime-detection, testing-and-verification |

## Workflow
1. Architect analyses the repo, writes specs/ADRs and the shared contracts (`contracts.py`, `interfaces.py`, `errors.py`, `core/events.py`).
2. Wave A (parallel, disjoint paths): data-engineering, runtime-compatibility, security (module), frontend-engineering.
3. Wave B: gpu-memory-optimization → llm-training; backend-engineering (parallel with training).
4. Architect integrates `studio.py` facade + `cli.py`.
5. qa-testing → security (review) → documentation → Architect final review.

See `docs/agent-workflow.md` for the task board and dependency graph.

## Rules
- Shared contracts change only through the Architect with an ADR.
- No simultaneous edits to the same file; out-of-scope changes are requested in the agent's report.
- Each contribution is reviewed by the Architect, who re-runs its tests before accepting ("review between agents").
- Done = verified: report test commands and output. Anything not verifiable here is reported as **UNVERIFIED** with the reason.
- Never: row caps, full-dataset materialisation, `trust_remote_code=True`, logging secrets, claiming public UI reachability without a probe.

## Dev environment
- Windows dev box, no GPU. Python venv: `.venv` (CPython 3.11). Run tests: `.venv/Scripts/python.exe -m pytest -q`.
- Installed: torch 2.15 (CPU), transformers 5.23, peft 0.21, trl 1.18, datasets 5.2, fastapi 0.162, pyarrow 26, polars 2.1, pydantic 2.14. Node 24 / npm 12.

## Ownership changes
- 2026-11-14: `core/lifecycle.py` moved from backend-engineering to chief-architect (it is the glue between the Studio facade and the training engine).
- 2026-11-14: Architect relaxed `security.validation.HF_REPO_ID_RE` to accept bare legacy ids (`gpt2`).
