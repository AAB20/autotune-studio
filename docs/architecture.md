# AutoTune Studio — Architecture

## Overview
```
 Browser tab (React SPA)  ──HTTPS──►  Colab proxy / cloudflared tunnel / localhost
        │  REST /api/*  +  WebSocket /ws (events)
        ▼
 FastAPI app (api/)  ── Studio facade (studio.py) ── EventBus (core/events.py)
        │                    │            │               ▲
        │      core/lifecycle (single training job, worker thread)
        │                    │            │               │ publish
  security/ (auth,      data_engine/   training/ ── monitoring/ (resources, OOM policy)
  path allowlist,       connectors/    export/
  redaction)
                 runs inside the notebook kernel process (uvicorn in a background thread)
```

## Module boundaries
| Package | Responsibility | Depends on |
|---|---|---|
| `contracts.py` | Pydantic models shared with the API and frontend (TS types generated) | pydantic |
| `interfaces.py` | Protocols: `Connector`, `StreamingDatasetProtocol`, `TrainingEngineProtocol`, `EventSink`, `StreamCursor`, `TrainingControl` | contracts |
| `errors.py` | `StudioError` hierarchy → `ApiError` codes | contracts |
| `core/` | environment + hardware detection, events, config, lifecycle | contracts |
| `connectors/` | source → Arrow record batches + cursor | pyarrow, ijson, (datasets) |
| `data_engine/` | discovery, streaming, batching, validation, memory budget → `StreamingDataset` | connectors |
| `monitoring/` | resource snapshots, sampler, metric store, OOM policy | psutil, (torch, pynvml) |
| `training/` | backend selection, model loading, LoRA/QLoRA, trainer loop, checkpoints | data_engine, monitoring, (torch, transformers, peft, trl, unsloth) |
| `export/` | adapter / merged export | training checkpoints |
| `security/` | session tokens, path allowlist, upload validation, log redaction | stdlib |
| `studio.py` | public Python API (`Studio`) composing everything | all of the above |
| `api/` | FastAPI routes + WebSocket | studio, security |
| `launch/` | uvicorn server thread, Colab proxy, cloudflared, local adapters | api, core |
| `frontend/` | React SPA built into `autotune_studio/web/` | REST/WS contracts |

Heavy ML dependencies are imported lazily so the API, data engine and UI work without them.

## Zero row configuration (ADR-0003)
Datasets are never row-capped or materialised. Connectors stream Arrow batches sized by a byte budget; discovery samples by bytes; cardinality is taken from metadata when available, otherwise **unknown** (`rows=None`), optionally resolved by an explicit full streaming scan. Training with unknown length requires `max_steps`; with known length, epochs are converted to steps.

## Concurrency
- uvicorn runs in a daemon thread inside the kernel; the notebook cell returns immediately.
- One training job at a time on a dedicated worker thread; control via `TrainingControl` events (stop / pause / save_now).
- `EventBus` is thread-safe; WebSocket handlers consume via asyncio queues and replay from a bounded buffer on reconnect.

## Security (ADR-0005)
Per-launch random session token, delivered in the URL fragment, exchanged for an HttpOnly cookie; every API/WS call authenticated; file access restricted to allowed roots; `trust_remote_code=False` everywhere; secrets redacted from logs.
