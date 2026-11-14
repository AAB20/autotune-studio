---
name: fastapi-backend
description: FastAPI app, routes, WebSocket event stream, task lifecycle, auth wiring and error mapping conventions. Load for api/, core/lifecycle.py, core/config.py.
---
# fastapi-backend

## Purpose
Expose the `Studio` facade over an authenticated HTTP + WebSocket API consumed by the React UI.

## Activation conditions
- Editing `autotune_studio/api/**`, `core/lifecycle.py`, `core/config.py`.

## Required inputs
- `contracts.py`, `errors.py`, `core/events.EventBus`, `studio.Studio`, `security/authentication.py`.

## Execution procedure
1. `api/app.create_app(studio, settings) -> FastAPI`; routers in `api/routes/*.py` under `/api`; WebSocket at `/ws` in `api/websocket.py`; SPA static files from `autotune_studio/web` with `index.html` fallback (only if built).
2. All `/api/*` except `GET /api/health` and `POST /api/auth/session`, plus `/ws`, require auth (security-audit skill).
3. Map `StudioError` → JSON `ApiError` with its `http_status`; Pydantic validation → 422 `validation_error`; unexpected → 500 `internal_error`, no traceback in the body, logged through the redaction filter.
4. Blocking work (dataset inspection, export, checkpoint save) runs via `starlette.concurrency.run_in_threadpool`; training runs on the lifecycle worker thread. Endpoints return promptly.
5. WS protocol: client connects `/ws?since=<seq>`; server sends `{"type":"hello","data":{"last_seq":N,"gap":bool}}`, then replayed + live `Event` JSON; server pings (`{"type":"ping"}` text is not an Event — use WebSocket ping or skip) every 20 s; client may send `{"type":"ack"}` (ignored).
6. One active training job; a second start → 409 `conflict`.

## Constraints
- Response models come from `contracts.py` — no ad-hoc dict schemas.
- No blocking calls on the event loop.

## Expected outputs
Routes:
`GET /api/health`, `POST /api/auth/session`, `GET /api/env`, `GET /api/hardware`, `GET /api/resources`,
`POST /api/datasets` (DatasetSpec → DatasetInfo), `POST /api/datasets/upload` (multipart),
`GET /api/datasets` (list), `GET /api/datasets/{id}`, `GET /api/datasets/{id}/preview?n=`, `GET /api/datasets/{id}/stats`,
`POST /api/datasets/{id}/scan` (full streaming pass for cardinality/validation), `POST /api/datasets/{id}/cancel`,
`GET/PUT /api/model`, `GET /api/models/suggested`, `GET/PUT /api/training/config`,
`POST /api/training/start|stop|pause|resume`, `GET /api/training/status`, `GET /api/training/metrics`,
`GET /api/sessions`, `GET /api/checkpoints`, `POST /api/checkpoints` (save now), `POST /api/checkpoints/{id}/resume`, `POST /api/export`.

## Validation checklist
- [ ] TestClient tests for every route incl. 401 without auth.
- [ ] WS test: events received; reconnect with `since` replays missed events; unauthenticated WS rejected.
- [ ] Error-mapping test (StudioError and unexpected exception).
