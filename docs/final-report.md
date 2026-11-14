# AutoTune Studio MVP — Final report (Chief Architect, 2026-11-14)

## Summary
A working MVP of AutoTune Studio: a Python package (`autotune_studio`) that runs a FastAPI backend inside a
notebook/script process, streams datasets with **zero row configuration**, fine-tunes LLMs with LoRA (hf
backend, verified on CPU) or QLoRA/Unsloth (GPU path, unverified here), and serves a React dashboard in a
separate browser tab, exposed through the Colab proxy, a cloudflared tunnel, or localhost.

Verification environment: Windows 11, **no GPU**, Python 3.11 venv (torch 2.15 CPU, transformers 5.23,
peft 0.21, trl 1.18, datasets 5.2), Node 24. Nothing has been run on Colab or Kaggle.

## Completed components
| Component | Location | Evidence |
|---|---|---|
| Shared contracts, interfaces, errors, event bus | `autotune_studio/{contracts,interfaces,errors}.py`, `core/events.py` | `tests/unit/test_imports.py`, TS types generated (`scripts/gen_ts_types.py`) |
| Runtime + hardware detection, launch adapters (Colab proxy, cloudflared, local) | `core/environment.py`, `core/hardware.py`, `launch/` | `tests/unit/runtime`, `tests/compatibility`; real cloudflared tunnel probe passed once on this machine |
| Zero-row data engine (CSV/JSON/JSONL/Parquet/HF streaming, discovery, validation, cursors, split) | `connectors/`, `data_engine/` | `tests/unit/data`, `tests/integration/test_dataset_scale.py` (150 MB files) |
| Training (hf + Unsloth engines), step planning, exact data-cursor checkpoints, OOM policy, export | `training/`, `export/`, `monitoring/` | `tests/unit/training`, `tests/integration/test_e2e_http_training.py` |
| Lifecycle + `Studio` facade + CLI | `core/lifecycle.py`, `studio.py`, `cli.py` | `tests/unit/core`, end-to-end smoke runs |
| FastAPI backend + WebSocket | `api/` | `tests/unit/api` |
| Security (auth, path policy, uploads, redaction, CSRF) + review | `security/`, `docs/security-review.md` | `tests/unit/security` |
| React dashboard (6 pages) | `frontend/`, built into `autotune_studio/web/` | vitest 54 passed; browser E2E (`docs/e2e-browser-check.md`) |
| Notebooks | `notebooks/colab.ipynb`, `notebooks/kaggle.ipynb` | nbformat-valid; **not executed** |
| Docs | `README.md`, `docs/*` | commands run where possible |
| Multi-agent config + skills | `AGENTS.md`, `.agents/agents` (10), `.agents/skills` (10), `scripts/sync_agents.py` | `tests/unit/test_agent_config.py` |

## Agent contributions
| Agent | Delivered | Defects found by others / fixed by Architect |
|---|---|---|
| chief-architect | contracts, ADR-0001..0006, lifecycle, Studio facade, CLI, integration fixes, final review | — |
| data-engineering | connectors, data engine | DEF-QA-01 corrupt parquet → 500 (fixed); SEC-07 HF token sources (fixed) |
| runtime-compatibility | detection, server thread, Colab proxy, cloudflared (pinned + SHA256), adapters | — |
| backend-engineering | routes, auth wiring, streaming multipart, WS, settings | settings not applied by launch (fixed) |
| frontend-engineering | dashboard SPA | stale list after scan, `#token=` in open tab (fixed) |
| llm-training | engines, checkpoints, resume, export | browser-connected training crash (fixed); SEC-06, SEC-09 (fixed) |
| gpu-memory-optimization | monitoring, sampler, metric store, ETA, OOM policy | — |
| qa-testing | conftest, integration + compatibility suites, acceptance matrix, progress report | — |
| security | security module, review SEC-01..12, npm audit | — |
| documentation | README, 7 docs, 2 notebooks, wheel verification | — |

## Skills created and used
`architecture-design`, `notebook-runtime-detection`, `streaming-data-engineering`, `dataset-validation`,
`llm-finetuning`, `gpu-memory-optimization`, `fastapi-backend`, `react-dashboard`, `security-audit`,
`testing-and-verification`. Each agent's dispatch required reading its listed SKILL.md files; every agent
report cited the skills it loaded. `scripts/sync_agents.py` validates them and mirrors them into `.claude/`.

## Tests executed (final run)
- Python: `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests` → **575 passed, 9 skipped** (153.8 s).
  Skips: opt-in network tests (`--run-network`), the opt-in tunnel smoke test, symlink tests (no privilege on
  this Windows account; junction variants run), GPU-only tests.
- Frontend: `npm test -- --run` → **54 passed**; `npm run typecheck` and `npm run build` pass.
- `npm audit` → 0 vulnerabilities.
- Browser E2E against the real backend (`docs/e2e-browser-check.md`): passed after fixing 3 defects it found.

## Defects found during integration (all fixed, each with a regression test)
1. Training crashed whenever a browser was connected (`datasets` fingerprinting the generator via dill reached
   WebSocket asyncio loops) — `tests/unit/training/test_train_stream_hashing.py`.
2. Lifecycle metric store replaced by an empty one (`metrics or MetricsStore()`) → metrics endpoint always `[]`.
3. Corrupt Parquet by path → 500 instead of 422 (DEF-QA-01).
4. SEC-01 (High): Hugging Face dataset names could read local files outside allowed roots.
5. SEC-04, SEC-06, SEC-02, SEC-07, SEC-09 (see `docs/security-review.md`).

## Acceptance status
`docs/acceptance.md`: **45 PASS / 0 FAIL / 5 UNVERIFIED** of 50 criteria (QA reconstructed the criteria from the
project docs; several PASS rows carry an UNVERIFIED GPU/real-runtime sub-part, stated per row).

## Runtime compatibility evidence
- Local Windows/CPU: full flow verified (detection, launch, UI, training, checkpoints, export).
- Colab / Kaggle / Jupyter: detection and adapter ordering verified only with simulated environments
  (`tests/compatibility/test_runtime_matrix.py`). cloudflared quick tunnel verified from this machine once.

## Known limitations / UNVERIFIED
See `docs/known-limitations.md`. Most important:
- **UNVERIFIED:** real Colab proxy, Kaggle networking, notebook execution, Unsloth/QLoRA/bitsandbytes on GPU,
  real CUDA OOM recovery, VRAM/NVML monitoring, Hugging Face Hub streaming, Python 3.10/3.12/3.13.
- One training job at a time; streams cannot be globally shuffled (bounded buffer); CSV reading adds up to
  ~235 MiB RSS (bounded); quick tunnels have no uptime guarantee; long merged exports through a tunnel may hit
  the proxy timeout; slow WebSocket clients can miss events beyond the replay buffer; no database connectors yet.
- The built UI is committed in `autotune_studio/web/` and must be rebuilt after frontend changes.

## How to launch
Local:
```
uv venv -p 3.11 .venv
uv pip install -p .venv --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch
uv pip install -p .venv -e ".[hf,train,dev]"
.venv/Scripts/python -m autotune_studio launch          # opens the UI in a new browser tab
```
Colab / Kaggle: open `notebooks/colab.ipynb` or `notebooks/kaggle.ipynb`, install from the repository git URL or
an uploaded wheel, then `Studio().launch()`. On Kaggle, turn on Internet in the notebook settings first.
