---
name: testing-and-verification
description: Test layout, acceptance criteria, PASS/FAIL/UNVERIFIED reporting and the no-claim-without-evidence rule. Load before writing tests or reporting completion.
---
# testing-and-verification

## Purpose
Every completion claim is backed by a test run whose output is quoted.

## Activation conditions
- Writing tests; finishing any task; updating the progress report.

## Required inputs
- Acceptance criteria in `docs/acceptance.md`.

## Execution procedure
1. Layout: `tests/unit/<area>/`, `tests/integration/`, `tests/compatibility/`. Shared fixtures in `tests/conftest.py` (synthetic dataset factories, tiny model factory).
2. Synthetic data generated in `tmp_path`; tiny models only; markers `slow`, `network`, `gpu`.
3. Commands (Windows dev box): `.venv/Scripts/python.exe -m pytest -q`; frontend: `npm test -- --run`, `npm run build` in `frontend/`.
4. Report each criterion as PASS (test id + evidence), FAIL (output) or UNVERIFIED (why it cannot run here + how to verify).
5. Update `docs/progress-report.md`.

## Constraints
- Never report PASS from reasoning alone.
- Never delete or skip failing tests to get green; fix them, or mark `xfail(reason=…)` and record an open item.

## Expected outputs
- Tests + updated acceptance matrix.

## Validation checklist
- [ ] Full suite run after integration, output captured.
- [ ] Every acceptance criterion has a status.
