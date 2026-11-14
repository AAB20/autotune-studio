# AutoTune Studio — Acceptance matrix

Owner: qa-testing. Last full run: 2026-11-14, Windows 11 dev box, CPython 3.11, torch CPU, **no GPU**.

Command: `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests`
Result: `573 passed, 9 skipped, 1 xfailed` (see `docs/progress-report.md` for the breakdown).

Status rules (skill `testing-and-verification`): **PASS** = a test in this repo ran and passed (test ids
given); **FAIL** = a test demonstrates the defect (strict `xfail` id given); **UNVERIFIED** = cannot be
exercised on this machine (reason + how to verify). Anything needing a GPU, a real Colab/Kaggle VM or a
real browser inside a notebook is UNVERIFIED unless evidence exists. "Simulated" means the runtime was
faked by monkeypatching (`google.colab` in `sys.modules`, `KAGGLE_KERNEL_RUN_TYPE`, fake proxy/tunnel).

The master spec text is not checked into the repository; the criteria below are reconstructed from the
spec categories (Runtime / Dataset / Interface / Training / Reliability), `AGENTS.md`,
`docs/architecture.md` and ADR-0001..0006. The Architect should reconcile wording with the spec.

Abbreviations: `I/` = `tests/integration/`, `C/` = `tests/compatibility/`, `U/` = `tests/unit/`.

## Runtime

| # | Criterion | Status | Evidence / how to verify |
|---|---|---|---|
| R1 | Installs and runs in Google Colab | UNVERIFIED | Needs a Colab VM. Verify: run `notebooks/` Colab notebook top to bottom; `studio.launch()` must print a proxy URL and the UI must open. |
| R2 | Installs and runs in Kaggle | UNVERIFIED | Needs a Kaggle kernel (Internet ON). Verify with the Kaggle notebook; expect `method="cloudflared"`, `publicly_verified=True`. |
| R3 | Runs in local Jupyter and plain Python scripts | PASS (script) / UNVERIFIED (real Jupyter kernel) | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down`, `C/test_launch_and_packaging.py::test_python_m_env_prints_valid_json`. Jupyter only simulated: `C/test_runtime_matrix.py::test_jupyter_and_script_detection`. Verify: `jupyter lab` -> `Studio().launch()`. |
| R4 | Runtime auto-detected (Colab / Kaggle / Jupyter / script) with correct workspace + allowed roots | PASS (simulated) | `C/test_runtime_matrix.py::test_colab_detection_roots_and_studio[module,env]`, `::test_kaggle_detection_roots[env,dir]`, `::test_jupyter_and_script_detection`, `::test_colab_wins_over_kaggle_and_jupyter`, `::test_extra_allowed_roots_env_on_every_runtime`, `U/runtime/test_environment.py`. Real VMs: see R1/R2. |
| R5 | Hardware auto-detected (CPU, RAM, disk, GPU, CUDA, bf16) | PASS (CPU/RAM/disk) / UNVERIFIED (GPU) | `U/runtime/test_hardware.py::test_real_machine`, `C/test_launch_and_packaging.py::test_python_m_env_prints_valid_json`; GPU paths only mocked (`U/runtime/test_hardware.py::test_fake_cuda_*`). Verify on a T4/P100: `python -m autotune_studio env` lists the GPU with VRAM. |
| R6 | Supported Python 3.10–3.13 (Colab/Kaggle ship 3.10–3.12) | PASS (metadata, 3.11 run) / UNVERIFIED (3.10, 3.12, 3.13 runs) | `C/test_launch_and_packaging.py::test_requires_python_covers_colab_kaggle_versions`. Suite executed only on 3.11. Verify: run the suite under 3.10 and 3.12 (`uv venv -p 3.10`). |
| R7 | Kaggle with Internet disabled gives actionable diagnostics, no tunnel attempt | PASS (simulated) | `C/test_runtime_matrix.py::test_kaggle_internet_off_diagnostics`, `U/runtime/test_adapters.py::test_kaggle_internet_disabled_diagnostic`. |
| R8 | Package installable with console entry point; imports without heavy ML deps | PASS | `C/test_launch_and_packaging.py::test_installed_distribution_has_console_entry_point`, `::test_cli_help_lists_commands`, `U/test_imports.py`. |
| R9 | Backend runs in a background thread; notebook cell returns immediately | PASS (script process) | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down` (launch returns, server answers over real HTTP), `U/runtime/test_server.py::test_start_from_running_event_loop`. Inside a real kernel: UNVERIFIED (see R3). |

## Dataset

| # | Criterion | Status | Evidence / how to verify |
|---|---|---|---|
| D1 | Hugging Face datasets streamed (`streaming=True`, no `trust_remote_code`) | UNVERIFIED | Network tests skipped by default. Verify: `pytest --run-network tests/unit/data/test_connectors.py` (Hub access). |
| D2 | Local CSV / JSON / JSONL / Parquet supported (file + upload) | PASS | `I/test_dataset_formats.py::test_format_matrix_load_preview_scan[*]` (11 combos), `::test_jsonl_content_with_json_extension_is_detected`, upload: `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload`. |
| D3 | Zero row configuration: no API, route or config field asks for a row count | PASS | `I/test_zero_row_config.py::test_facade_signatures_have_no_row_count_parameters`, `::test_contract_models_have_no_row_count_fields`, `::test_no_route_requires_a_row_count`. |
| D4 | Unknown cardinality reported as unknown; exact after scan; Parquet exact from metadata | PASS | `I/test_dataset_scale.py::test_scan_large_file_exact_count_stable_checksum_bounded_rss[*]`, `I/test_zero_row_config.py::test_parquet_cardinality_known_from_metadata`, `I/test_dataset_formats.py::test_background_scan_via_http_reaches_exact_cardinality`. |
| D5 | Unknown length -> training requires `max_steps` with clear 422 `max_steps_required` | PASS | `I/test_zero_row_config.py::test_unknown_cardinality_without_max_steps_is_422_steps_required`. |
| D6 | Large files streamed with bounded memory; every row counted; checksum stable across passes | PASS (150 MB JSONL, 150 MB CSV) | `I/test_dataset_scale.py::test_scan_large_file_exact_count_stable_checksum_bounded_rss[jsonl-…]` (RSS growth < 50 % of size; measured ~65 MiB for 150–300 MB), `[csv-…]` (RSS growth < engine memory budget + 64 MiB; pyarrow CSV has a ~150–235 MiB fixed overhead, see progress report), `U/data/test_large_memory.py`. Datasets larger than RAM: UNVERIFIED (not attempted). |
| D7 | Format auto-detected (chat / instruction / text) and text columns chosen | PASS | `I/test_dataset_formats.py::test_format_matrix_load_preview_scan[*]`. |
| D8 | Malformed rows counted and reported, never silently dropped | PASS | `I/test_failure_recovery.py::test_malformed_jsonl_lines_are_counted_not_dropped`, `::test_all_malformed_jsonl_is_rejected_not_silently_empty`. |
| D9 | Empty dataset -> `dataset_empty`; corrupt file -> `dataset_malformed` (422, never 500) | PASS (DEF-QA-01 fixed by Architect) |  PASS: `I/test_failure_recovery.py::test_empty_file_is_dataset_empty[*]`, `::test_corrupt_parquet_upload_is_422_dataset_malformed`. FAIL: `I/test_failure_recovery.py::test_corrupt_parquet_by_path_is_422_dataset_malformed` (strict xfail, DEF-QA-01). |
| D10 | Dataset files are never modified by load / preview / scan / train / export | PASS | sha256 before/after in `I/test_e2e_http_training.py::*`, `I/test_dataset_scale.py::*`, `I/test_dataset_formats.py::*`, `I/test_failure_recovery.py::*`. |
| D11 | Paths restricted to allowed roots | PASS | `I/test_failure_recovery.py::test_missing_and_forbidden_paths`, `U/security/test_validation.py`. HF-source path bypass (SEC-01): `U/security/test_review_findings.py::test_sec01_*` now pass. |
| D12 | Streams resumable from the checkpointed cursor | PASS | `I/test_e2e_http_training.py::test_stop_mid_run_then_resume_from_checkpoint` (resumed job starts at checkpoint step + 1), `U/training/test_checkpoints_data.py::test_train_stream_multi_pass_and_exact_resume`. |

## Interface

| # | Criterion | Status | Evidence / how to verify |
|---|---|---|---|
| I1 | UI opens in a separate browser tab | PASS (local link) / UNVERIFIED (notebook) | The launch link opened the UI in its own browser tab and authenticated (`docs/e2e-browser-check.md`). The notebook link/`window.open` behaviour in real Colab/Kaggle/JupyterLab is UNVERIFIED. |
| I2 | Built SPA served by the backend (index, assets, history fallback) | PASS | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down`, `::test_built_ui_assets_referenced_by_index_are_served`. |
| I3 | UI pages: dataset, model, training dashboard, live charts, checkpoints/export | PASS (local browser) | Architect browser E2E 2026-11-14 (`docs/e2e-browser-check.md`): token bootstrap, dataset load + scan, model, training start with inline max_steps validation, live loss/LR charts over WS, checkpoints, LoRA export. Frontend unit tests: 54 passed (`frontend/src/test/`). |
| I4 | Session token only in URL fragment (`#token=`), never `?token=`; exchanged for HttpOnly cookie | PASS | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down`, `C/test_runtime_matrix.py::test_default_adapter_order_per_runtime[*]`, `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload` (cookie flow). |
| I5 | All API/WS routes authenticated except health + session exchange | PASS | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down` (401 over real HTTP), `U/api/test_auth.py::test_protected_routes_require_auth`, `U/api/test_websocket.py::test_unauthenticated_ws_rejected`. |
| I6 | Live updates over WebSocket with increasing `seq` (status / metric / checkpoint events) | PASS | `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload`. |
| I7 | WS reconnect with `since` replays missed events: no duplicates, no gaps | PASS | `I/test_e2e_http_training.py::test_ws_disconnect_reconnect_no_duplicates_no_gaps`. |
| I8 | Colab exposure via kernel proxy; Kaggle via cloudflared; local fallback | PASS (simulated) / UNVERIFIED (real) | `C/test_runtime_matrix.py::test_default_adapter_order_per_runtime[*]`, `::test_studio_launch_on_simulated_colab_uses_proxy`, `::test_tunnel_failure_falls_back_to_local_with_diagnostics`. Real tunnel: `AUTOTUNE_TUNNEL_SMOKE=1 pytest --run-network U/runtime/test_tunnel_smoke.py`; real Colab proxy: R1. |
| I9 | `publicly_verified=True` only after a successful health probe through the public URL | PASS (simulated) | `C/test_runtime_matrix.py::test_publicly_verified_only_with_passing_probe[True,False]`; Colab proxy always unverified: `::test_default_adapter_order_per_runtime[colab-…]`. |
| I10 | Public tunnel origin accepted for WS; cross-origin rejected | PASS (simulated) | `C/test_runtime_matrix.py::test_studio_launch_on_simulated_colab_uses_proxy`, `U/api/test_websocket.py::test_cross_origin_rejected_even_with_token`. |

## Training

| # | Criterion | Status | Evidence / how to verify |
|---|---|---|---|
| T1 | LoRA fine-tuning runs end-to-end via the UI API (hf backend) | PASS (CPU, tiny model) | `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload`. |
| T2 | QLoRA 4-bit / Unsloth backend on CUDA | UNVERIFIED | Needs CUDA + bitsandbytes + unsloth. Mocked only (`U/training/test_unsloth_and_network.py`). Verify on Colab T4 with `method="qlora"`. |
| T3 | Live metrics (loss, lr) available via REST + WS | PASS | `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload` (`GET /api/training/metrics`, metric events). |
| T4 | Stop mid-run saves a checkpoint; state `stopped` | PASS | `I/test_e2e_http_training.py::test_stop_mid_run_then_resume_from_checkpoint`. |
| T5 | Save-now checkpoint while running; pause/resume | PASS | save-now via HTTP: `I/test_e2e_http_training.py::test_stop_mid_run_then_resume_from_checkpoint`; pause: `U/training/test_engine.py::test_pause_save_now_and_resume`, `U/training/test_lifecycle_integration.py`. |
| T6 | Resume from checkpoint continues at the saved step to completion | PASS | `I/test_e2e_http_training.py::test_stop_mid_run_then_resume_from_checkpoint` (`POST /api/checkpoints/{id}/resume`). |
| T7 | Export LoRA adapter; reloadable with PEFT | PASS | `I/test_e2e_http_training.py::test_full_http_flow_train_export_reload`. |
| T8 | Export merged 16-bit model; reloadable with Transformers | PASS (CPU, tiny model) | same test. Merged export of a 4-bit/Unsloth model: UNVERIFIED (GPU). |
| T9 | Epochs converted to steps when length known | PASS | `I/test_zero_row_config.py::test_unknown_cardinality_without_max_steps_is_422_steps_required` (after scan), `U/training/test_planning.py`. |
| T10 | CUDA OOM handled: halve micro-batch, keep effective batch, resume from last checkpoint | PASS (simulated OOM) / UNVERIFIED (real GPU) | `U/training/test_engine.py::test_simulated_oom_recovers_from_last_checkpoint`, `U/monitoring/test_oom.py`. Verify on a GPU with an oversized batch. |
| T11 | Resource monitoring (CPU/RAM/VRAM) streamed | PASS (CPU/RAM) / UNVERIFIED (VRAM) | `resource` events observed in `I/test_e2e_http_training.py` runs; `U/monitoring/test_resources.py`. |
| T12 | `trust_remote_code` never enabled | PASS | `U/api/test_training.py::test_model_invalid_names_422`, `ModelConfig.trust_remote_code: Literal[False]`. |

## Reliability

| # | Criterion | Status | Evidence / how to verify |
|---|---|---|---|
| L1 | Client/data errors map to typed 4xx JSON, never 500 with traceback | PASS (DEF-QA-01 fixed) | `I/test_failure_recovery.py::test_corrupt_parquet_by_path_is_422_dataset_malformed` (xfail), `::test_bad_ids_and_idle_controls_are_4xx`, `U/api/test_app_misc.py`. |
| L2 | Training failure -> state `failed` + `error` event; server keeps working; next job starts | PASS | `I/test_failure_recovery.py::test_training_failure_reported_and_server_recovers[repo,empty_dir]`. |
| L3 | Only one training job at a time (409 conflict) | PASS | `U/api/test_training.py::test_second_start_conflict_and_controls`, `I/test_failure_recovery.py::test_bad_ids_and_idle_controls_are_4xx` (save-now with no job = 409). |
| L4 | Clean shutdown (server, tunnel, sampler) | PASS | `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down` (health unreachable after `shutdown()`). |
| L5 | Secrets never in logs / API responses / events | PASS | `U/security/test_review_findings.py::test_sec04_exception_text_with_token_not_exposed`, `::test_sec04_redact_data_unit` (SEC-04 fixed during this wave; formerly xfail), `U/security/test_redaction_and_secrets.py`, launch info: `C/test_launch_and_packaging.py::test_launch_local_serves_api_ui_and_shuts_down`. |
| L6 | Long-running sessions survive notebook-proxy idle timeouts (keepalive) | PASS (unit) / UNVERIFIED (real proxy) | `U/api/test_websocket.py::test_keepalive_ping`. Verify on Colab: leave UI open > 10 min during training. |
| L7 | Colab runtime recycle / Drive persistence of checkpoints | UNVERIFIED | Needs Colab with Drive mounted. Detection note covered by `C/test_runtime_matrix.py::test_colab_detection_roots_and_studio`. |

## Summary

| Status | Count (of 50 criteria) |
|---|---|
| PASS (incl. "PASS (simulated)" and rows that are PASS here but UNVERIFIED on GPU/real VM for a sub-part) | 45 |
| FAIL | 0 (DEF-QA-01 fixed) |
| UNVERIFIED (entirely) | 5 (R1, R2, D1, T2, L7) |

Counting rule: a row with a split status counts toward its first status (e.g. "PASS (CPU) / UNVERIFIED (GPU)" counts
as PASS; its UNVERIFIED part is listed in the row). Rows with partial UNVERIFIED parts: R3, R5, R6, R9, D6, I8, T8, T10, T11, L6.
