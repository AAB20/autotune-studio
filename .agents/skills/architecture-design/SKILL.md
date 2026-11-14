---
name: architecture-design
description: Architecture standards, module boundaries, dependency rules and coding conventions for AutoTune Studio. Load before creating modules, changing contracts or adding dependencies.
---
# architecture-design

## Purpose
Keep AutoTune Studio modular: packages talk to each other only through `autotune_studio/contracts.py` (Pydantic data), `autotune_studio/interfaces.py` (Protocols), `autotune_studio/errors.py` and `autotune_studio/core/events.py` (EventBus).

## Activation conditions
- Creating a new module/package or public function.
- Changing anything in `contracts.py`, `interfaces.py`, `errors.py`, `core/events.py`, `pyproject.toml`.
- Adding or upgrading a dependency.

## Required inputs
- Task description and owned paths (`AGENTS.md` ownership map).
- `docs/architecture.md`, ADRs in `docs/decisions/`.

## Execution procedure
1. Read `AGENTS.md`, `docs/architecture.md`, `contracts.py`, `interfaces.py`.
2. Confirm the change lies within your owned paths; otherwise ask the Architect (coordinator) to route it to the owner.
3. Dependency direction (no upward imports):
   `contracts/interfaces/errors` ← `core` ← `connectors` ← `data_engine` ← `monitoring` ← `training`/`export` ← `studio.py` ← `api` ← `launch`/`cli`.
4. Heavy optional deps (`torch`, `transformers`, `datasets`, `peft`, `trl`, `unsloth`, `pynvml`) are imported lazily inside functions and raise `errors.DependencyMissing` with an install hint when absent.
5. Contract changes require an ADR (`docs/decisions/ADR-NNNN-*.md`) and regeneration of TS types (`python scripts/gen_ts_types.py`).
6. Installed versions are the compatibility truth: check the real API of the installed package (`python -c "import inspect, X; ..."`) instead of relying on memory. Current dev env: torch 2.15 (CPU), transformers 5.23, peft 0.21, trl 1.18, datasets 5.2, fastapi 0.162, pyarrow 26, polars 2.1, pydantic 2.14.

## Constraints
- Python 3.10–3.13, full type hints, docstrings on public API.
- No global mutable state except the `Studio` instance and its `EventBus`.
- Do not reimplement Unsloth kernels; do not add frameworks outside ADR-0001 without a new ADR.

## Expected outputs
- Code inside owned paths, tests, and (if contracts changed) an ADR.

## Validation checklist
- [ ] `python -c "import autotune_studio, autotune_studio.studio"` works without importing torch at module import time.
- [ ] `tests/unit/test_imports.py` passes (no cycles, lazy heavy deps).
- [ ] `ruff check` clean on changed files.
- [ ] ADR written for any contract or dependency change.
