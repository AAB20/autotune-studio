# API reference

This reference is derived from `autotune_studio/studio.py`, `autotune_studio/api/` and
`autotune_studio/contracts.py` (version 0.1.0). Field-level schemas of the models are in
[configuration.md](configuration.md) and `contracts.py`.

## Python: `autotune_studio.Studio`

```python
from autotune_studio import Studio
```

Exceptions live in `autotune_studio.errors`. All of them subclass `StudioError(message, detail)`,
which carries `.code` and `.http_status` (see [error codes](#error-codes)). Pydantic
`ValidationError` is raised for invalid config field values.

### Construction and runtime

| Method | Returns | Notes |
|---|---|---|
| `Studio(workspace_dir=None, allowed_roots=None, *, resource_interval=2.0, replay_size=None)` | | Detects the runtime. Creates the workspace directory. Generates the session token. `allowed_roots` are added to the detected roots, and the workspace is always allowed. `replay_size` defaults to `AUTOTUNE_REPLAY_SIZE` (2000). |
| `detect_environment(refresh=False)` | `EnvironmentInfo` | `kind` (`colab`/`kaggle`/`jupyter`/`script`), `workspace_dir`, `allowed_roots`, `internet_access`, `notes`. `refresh=True` re-detects and probes the internet. |
| `detect_hardware(refresh=False)` | `HardwareInfo` | CPU, RAM, disk, `cuda_available`, `gpus`, `bf16_supported`, `notes`. Never raises. |
| `resources()` | `ResourceSnapshot` | Current CPU/RAM/VRAM usage. |
| `runs_dir` (property) | `Path` | `<workspace>/runs` |

### Datasets

| Method | Returns | Raises |
|---|---|---|
| `load_dataset(source, name=None, path=None, *, split="train", config_name=None, revision=None, text_column=None)` | `DatasetInfo` | `ForbiddenPath`, `NotFound`, `DatasetEmpty`, `DatasetMalformed`, `ValidationFailed` |
| `load_dataset_spec(spec: DatasetSpec)` | `DatasetInfo` | same |
| `list_datasets()` | `list[DatasetInfo]` | |
| `get_dataset(dataset_id=None)` | `StreamingDataset` handle (active dataset by default) | `NotFound` |
| `inspect_dataset(dataset_id=None, n=20)` | `DatasetInspection` (`info`, `preview`, `stats`) | `NotFound` |
| `dataset_stats(dataset_id=None)` | `StreamStats` | `NotFound` |
| `scan_dataset(dataset_id=None, background=False)` | `StreamStats` | `Conflict` (a background scan is already running) |
| `cancel_dataset(dataset_id=None)` | `StreamStats` | `NotFound` |

- `source` is one of `huggingface` (needs `name`, a Hub id) or `csv`, `json`, `jsonl`, `parquet`
  (need `path`, which must be inside an allowed root; a directory is allowed only for `parquet`).
- No row count is ever accepted. `DatasetInfo.cardinality.rows is None` means unknown.
  `method` is `metadata`, `scan` or `unknown`. A synchronous `scan_dataset()` streams every row
  once and makes the count exact.
- The most recently loaded dataset becomes the *active* dataset, used when `dataset_id` is omitted.

### Model and training configuration

| Method | Returns | Raises |
|---|---|---|
| `configure_model(model_name, **kwargs)` | `ModelConfig` | `ValidationFailed`, `ForbiddenPath` |
| `suggested_models()` | `list[str]` | |
| `configure_training(method="qlora", **kwargs)` | `TrainingConfig` | `ValidationError` |
| `set_training_config(cfg: TrainingConfig)` | `TrainingConfig` | |
| `validate_training(dataset_id=None)` | `(planned_steps: int, warnings: list[str])` | `ValidationFailed` (no model; QLoRA or 8-bit optimizer without CUDA; bf16 unsupported), `DependencyMissing`, `StepsRequired`, `DatasetEmpty`, `NotFound` |

- `model_name` is a Hugging Face repo id or a local directory inside an allowed root. URLs and
  `..` are rejected. `kwargs` are `ModelConfig` fields (`max_seq_length`, `load_in_4bit`,
  `dtype`, `revision`). `trust_remote_code` is always `False`.
- `configure_training` merges `kwargs` into the current config. Fields you do not pass keep
  their values, while `method` is always set (default `"qlora"`). To clear a field, pass it
  explicitly, for example `max_steps=None`.
- `model_config` and `training_config` are readable attributes.

### Training control

| Method | Returns | Raises / notes |
|---|---|---|
| `start_training(dataset_id=None, resume_from=None)` | `TrainingStatus` | `validate_training` errors, plus `Conflict` if a job is active. Runs in a background thread. `resume_from` is a checkpoint id or `CheckpointInfo`. The output directory defaults to `<workspace>/runs/<job_id>`. |
| `training_status()` | `TrainingStatus` | `state`: idle, preparing, running, paused, stopping, completed, stopped, failed |
| `training_metrics(since_step=None)` | `list[MetricPoint]` | |
| `stop_training(wait=True, timeout=120.0)` | `TrainingStatus` | Stops after the current step. A checkpoint is saved first. |
| `pause_training()` / `resume_training()` | `TrainingStatus` | |
| `wait(timeout=None)` | `TrainingStatus` | Blocks until the job finishes. |
| `sessions()` | `list[SessionSummary]` | History, persisted in `<workspace>/sessions.json`. |

### Checkpoints and export

| Method | Returns | Raises / notes |
|---|---|---|
| `save_checkpoint(timeout=300.0)` | `CheckpointInfo` | Saved at the next step boundary. `Conflict` if no job is running. |
| `list_checkpoints()` | `list[CheckpointInfo]` | All checkpoints under `runs/`, oldest first. |
| `resume_from_checkpoint(checkpoint_id, dataset_id=None)` | `TrainingStatus` | New job. Model and training config are restored from the checkpoint's `autotune_state.json`. |
| `export_model(checkpoint_id=None, format="lora_adapter", output_dir=None)` | `ExportResult` (`path`, `format`, `files`, `size_bytes`) | `format`: `lora_adapter` or `merged_16bit`. Uses the latest checkpoint by default. Default destination: `<workspace>/exports/<checkpoint_id>-<suffix>`. `Conflict` while training runs, unless you pass an explicit `checkpoint_id`. `NotFound` if there are no checkpoints. |

### Web UI

| Method | Returns | Notes |
|---|---|---|
| `launch(*, port=None, host="127.0.0.1", open_browser=True, prefer=None, tunnel_timeout=30.0)` | `LaunchInfo` | Starts uvicorn in a daemon thread, exposes it, shows or opens the link in a new tab. `prefer`: `colab_proxy`, `cloudflared`, `local` or `None` (the runtime default). Calling it again while running returns the same info and shows the link again. Raises `RuntimeError` if the backend cannot start. |
| `launch_info` (property) | `LaunchInfo \| None` | |
| `create_app()` | FastAPI app | For tests or embedding. |
| `shutdown(stop_training=True)` | `None` | Stops training (with a checkpoint), the tunnel, the server and the sampler. |

`LaunchInfo` fields: `local_url`, `public_url`, `method`, `publicly_verified` (True only after a
successful `GET <public_url>/api/health`), `open_url` (`<base>/#token=<token>`, a secret) and
`diagnostics`.

## REST API

Base: the launch URL. All bodies are JSON. Except the two public routes, every route requires one
of:

- the session cookie `autotune_session` (HttpOnly, SameSite=Strict, Secure on https). The SPA
  gets it from `POST /api/auth/session`. State-changing requests authenticated by cookie must also
  send a matching `Origin` (CSRF defence).
- `Authorization: Bearer <token>`, where the token is the `#token=` value from the launch link.

A missing or invalid credential returns `401 unauthorized` with `WWW-Authenticate: Bearer`. There
are no `/docs`, `/redoc` or `/openapi.json` routes.

| Method & path | Request | Response | Notes |
|---|---|---|---|
| `GET /api/health` | | `{"status":"ok","version":str}` | **public** |
| `POST /api/auth/session` | `{"token": str}` | `{"ok": true}` + cookie | **public**. 20 failed attempts per minute per client IP, then `429 rate_limited` with `Retry-After`. |
| `GET /api/env` | | `EnvironmentInfo` | |
| `GET /api/hardware?refresh=bool` | | `HardwareInfo` | |
| `GET /api/resources` | | `ResourceSnapshot` | |
| `GET /api/launch` | | `LaunchInfo \| null` | `open_url` is always `null`. |
| `POST /api/datasets` | `DatasetSpec` | `DatasetInfo` | |
| `POST /api/datasets/upload` | multipart, field `file` | `DatasetInfo` | Extensions: `.csv .tsv .json .jsonl .ndjson .parquet .pq`. Maximum `AUTOTUNE_UPLOAD_MAX_BYTES` (2 GiB). Stored as `<workspace>/uploads/<random>.<ext>`, and deleted if loading fails. |
| `GET /api/datasets` | | `list[DatasetInfo]` | |
| `GET /api/datasets/{id}` | | `DatasetInfo` | |
| `GET /api/datasets/{id}/preview?n=20` | | `DatasetPreview` | `0 <= n <= 1000` |
| `GET /api/datasets/{id}/stats` | | `StreamStats` | |
| `POST /api/datasets/{id}/scan` | | `StreamStats` | Background scan. Progress arrives as `dataset` events. |
| `POST /api/datasets/{id}/cancel` | | `StreamStats` | |
| `GET /api/model` | | `ModelConfig` | `404` until configured |
| `PUT /api/model` | `ModelConfig` | `ModelConfig` | |
| `GET /api/models/suggested` | | `list[str]` | |
| `GET /api/training/config` | | `TrainingConfig` | |
| `PUT /api/training/config` | `TrainingConfig` | `TrainingConfig` | Replaces the whole config for the next job. |
| `POST /api/training/start` | `{"dataset_id"?: str, "resume_from"?: str}` (optional body) | `TrainingStatus` | `409` if a job is active. `422 max_steps_required` when the length is unknown. |
| `POST /api/training/stop` | | `TrainingStatus` | Returns at once (`stopping`). |
| `POST /api/training/pause` / `resume` | | `TrainingStatus` | |
| `GET /api/training/status` | | `TrainingStatus` | |
| `GET /api/training/metrics?since_step=` | | `list[MetricPoint]` | |
| `GET /api/training/validate?dataset_id=` | | `{"planned_steps": int, "warnings": [str]}` | |
| `GET /api/sessions` | | `list[SessionSummary]` | |
| `GET /api/checkpoints` | | `list[CheckpointInfo]` | |
| `POST /api/checkpoints` | | `CheckpointInfo` | Saves now. Waits at most 90 s, then returns `409`. |
| `POST /api/checkpoints/{id}/resume` | `{"dataset_id"?: str}` (optional) | `TrainingStatus` | |
| `POST /api/export` | `ExportRequest` (`checkpoint_id?`, `format`, `output_dir?`) | `ExportResult` | `output_dir` must be inside an allowed root. |
| `GET /{path}` | | SPA `index.html` / assets | A "UI not built" page if `autotune_studio/web/` is empty. |

Example with Bearer auth (the `TOKEN` value is the `#token=` part of the launch link):

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7860/api/env
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"source":"jsonl","path":"data/train.jsonl"}' http://127.0.0.1:7860/api/datasets
```

The equivalent Bearer request with httpx was run against an installed wheel (`/api/env` → 200;
without the header → 401).

### Error envelope

Every error is JSON `ApiError`: `{"code": str, "message": str, "detail": object | null}`.
Request validation errors are `422 validation_error` with
`detail.errors = [{"loc": [...], "msg": str, "type": str}]`. Unhandled errors are
`500 internal_error` with `detail.error_id`; the traceback is only logged (redacted).

### Error codes

From `contracts.ErrorCode` and `errors.py`:

| Code | HTTP | Exception | Typical cause |
|---|---|---|---|
| `validation_error` | 422 | `ValidationFailed` | Bad field value. QLoRA or 8-bit optimizer without CUDA. No model configured. Bad upload. |
| `not_found` | 404 | `NotFound` | Unknown dataset/checkpoint id, no model configured (`GET /api/model`), no checkpoints, unknown route |
| `conflict` | 409 | `Conflict` | A training job is already active. Save-now without a running job. Export while training without a `checkpoint_id`. Scan already running. |
| `forbidden_path` | 403 | `ForbiddenPath` | Path outside the allowed roots, or an extension not allowed |
| `unauthorized` | 401 | `Unauthorized` | Missing/invalid token or cookie, or a cross-origin cookie request |
| `rate_limited` | 429 | `RateLimited` | Too many failed `POST /api/auth/session` attempts (`detail.retry_after_seconds`) |
| `dependency_missing` | 424 | `DependencyMissing` | Training stack, `unsloth` or `bitsandbytes` not installed (the message has the install hint) |
| `dataset_empty` | 422 | `DatasetEmpty` | No rows |
| `dataset_malformed` | 422 | `DatasetMalformed` | Corrupt or unparsable file, or every record malformed |
| `max_steps_required` | 422 | `StepsRequired` | Unknown dataset length (or `num_epochs` unset) and no `max_steps` |
| `out_of_memory` | 507 | `OutOfMemory` | OOM that could not be recovered (the micro-batch is already 1) |
| `internal_error` | 500 | other | Bug. Report it with `detail.error_id` |

The API layer also emits `method_not_allowed` (405), `payload_too_large` (413) and `http_<status>`
for other Starlette HTTP errors.

## WebSocket `/ws`

1. Connect to `ws(s)://<host>/ws?since=<seq>`. `since` is the highest `seq` you have processed,
   or 0. Credentials are the same as REST (cookie or Bearer). Browsers must send an `Origin` equal
   to the host or the public launch origin. Authentication runs before `accept`: a failure closes
   with code **1008**.
2. The first message is `{"seq":0,"type":"hello","ts":...,"data":{"last_seq":N,"gap":bool}}`.
   `gap=true` means events after `since` were evicted from the replay buffer
   (`AUTOTUNE_REPLAY_SIZE`); refetch REST state. If `since > last_seq`, the server restarted and
   the whole buffer is replayed from 0.
3. Next come the replayed events, then live events. Each one is an `Event`:
   `{"seq": int, "type": str, "ts": float, "data": {...}}`, with `seq` strictly increasing.
4. Every `AUTOTUNE_WS_PING_INTERVAL` seconds (20) the server sends `{"type":"ping","ts":...}`. This
   is not an `Event`; ignore it.
5. Messages from the client are read and ignored.

| `type` | `data` |
|---|---|
| `status` | `TrainingStatus` |
| `metric` | `MetricPoint` (`step`, `loss`, `learning_rate`, `grad_norm`, `epoch`, `eval_loss`, `timestamp`) |
| `resource` | `ResourceSnapshot` |
| `log` | `{"level": "debug\|info\|warning\|error", "source": str, "message": str, ...}` |
| `dataset` | `{"dataset_id": str, **StreamStats}` |
| `checkpoint` | `CheckpointInfo` |
| `export` | `ExportResult` |
| `error` | `{"source": "training"\|"dataset"\|..., "message": str, ...context (job_id, dataset_id, code)}` |

Payloads pass through secret redaction before they are published. A slow client's queue holds
1000 events and drops the oldest when full. The dropped events are not flagged until the client
reconnects with `since`.
