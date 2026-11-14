"""FastAPI application factory: a thin authenticated layer over the :class:`~autotune_studio.studio.Studio` facade.

Route table (all under ``/api`` require auth except ``health`` and ``auth/session``)::

    GET  /api/health                     HealthResponse          public
    POST /api/auth/session               {"ok": true} + cookie   public, rate-limited
    GET  /api/env                        EnvironmentInfo
    GET  /api/hardware?refresh=          HardwareInfo
    GET  /api/resources                  ResourceSnapshot
    GET  /api/launch                     LaunchInfo | null       (open_url always null)
    POST /api/datasets                   DatasetSpec -> DatasetInfo
    POST /api/datasets/upload            multipart 'file' -> DatasetInfo
    GET  /api/datasets                   list[DatasetInfo]
    GET  /api/datasets/{id}              DatasetInfo
    GET  /api/datasets/{id}/preview?n=   DatasetPreview (0..1000)
    GET  /api/datasets/{id}/stats        StreamStats
    POST /api/datasets/{id}/scan         StreamStats (background scan)
    POST /api/datasets/{id}/cancel       StreamStats
    GET  /api/model                      ModelConfig (404 until configured)
    PUT  /api/model                      ModelConfig -> ModelConfig
    GET  /api/models/suggested           list[str]
    GET  /api/training/config            TrainingConfig
    PUT  /api/training/config            TrainingConfig -> TrainingConfig
    POST /api/training/start             {dataset_id?, resume_from?} -> TrainingStatus (409 if active)
    POST /api/training/stop|pause|resume TrainingStatus
    GET  /api/training/status            TrainingStatus
    GET  /api/training/metrics?since_step=  list[MetricPoint]
    GET  /api/training/validate?dataset_id= {planned_steps, warnings}
    GET  /api/sessions                   list[SessionSummary]
    GET  /api/checkpoints                list[CheckpointInfo]
    POST /api/checkpoints                CheckpointInfo (save now)
    POST /api/checkpoints/{id}/resume    {dataset_id?} -> TrainingStatus
    POST /api/export                     ExportRequest -> ExportResult
    WS   /ws?since=<seq>                 event stream (see api/websocket.py)
    GET  /{path}                         SPA (index.html fallback) or "UI not built" page
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..core.config import StudioSettings, load_settings
from ..security import FailedAuthLimiter
from .deps import require_auth
from .errors import InternalErrorMiddleware, install_error_handlers
from .routes import auth, checkpoints, datasets, hub, models, system, training
from .static import DEFAULT_WEB_DIR, SecurityHeadersMiddleware, spa_handler
from .websocket import router as ws_router

if TYPE_CHECKING:  # pragma: no cover
    from ..studio import Studio

log = logging.getLogger("autotune_studio.api")

__all__ = ["create_app"]


def create_app(
    studio: Studio,
    settings: StudioSettings | None = None,
    *,
    web_dir: str | Path | None = None,
    auth_limiter: FailedAuthLimiter | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to ``studio``.

    :param settings: server settings; default :func:`~autotune_studio.core.config.load_settings` (env).
    :param web_dir: built SPA directory (default ``autotune_studio/web``).
    :param auth_limiter: failed-login limiter for ``POST /api/auth/session`` (injectable for tests).
    """
    settings = settings or load_settings()
    app = FastAPI(
        title="AutoTune Studio",
        version=__version__,
        docs_url=None,  # interactive docs would be an unauthenticated surface on a public tunnel
        redoc_url=None,
        openapi_url=None,
    )
    app.state.studio = studio
    app.state.settings = settings
    app.state.auth_limiter = auth_limiter or FailedAuthLimiter()

    install_error_handlers(app)

    app.include_router(auth.router, prefix="/api")
    protected = [Depends(require_auth)]
    for module in (system, datasets, models, training, checkpoints, hub):
        app.include_router(module.router, prefix="/api", dependencies=protected)
    app.include_router(ws_router)

    serve = spa_handler(Path(web_dir) if web_dir is not None else DEFAULT_WEB_DIR)
    app.add_api_route("/", serve, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route("/{path:path}", serve, methods=["GET", "HEAD"], include_in_schema=False)

    # Middleware: last added = outermost. Security headers wrap the 500 handler so error responses
    # carry them too; CORS (opt-in) is outermost so preflights are answered before anything else.
    app.add_middleware(InternalErrorMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    if settings.cors_enabled and settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,  # cookies are SameSite=Strict anyway; cross-origin clients use Bearer
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["Authorization", "Content-Type"],
        )
    # settings.replay_size is consumed where the EventBus is created (Studio); see backend report.
    return app
