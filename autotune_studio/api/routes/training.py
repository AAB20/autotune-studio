"""Training configuration, control and history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from ...contracts import MetricPoint, SessionSummary, TrainingConfig, TrainingStatus
from ..deps import get_studio
from ..schemas import StartTrainingRequest, TrainingValidation

router = APIRouter(tags=["training"])


@router.get("/training/config", response_model=TrainingConfig)
async def get_training_config(studio=Depends(get_studio)) -> TrainingConfig:
    return studio.training_config


@router.put("/training/config", response_model=TrainingConfig)
async def put_training_config(cfg: TrainingConfig, studio=Depends(get_studio)) -> TrainingConfig:
    """Replace the hyper-parameters used by the *next* job (a running job keeps its own copy)."""
    return studio.set_training_config(cfg)


@router.post("/training/start", response_model=TrainingStatus)
async def start_training(body: StartTrainingRequest | None = None, studio=Depends(get_studio)) -> TrainingStatus:
    """Start the single training job (409 ``conflict`` if one is already active)."""
    req = body or StartTrainingRequest()
    return await run_in_threadpool(
        lambda: studio.start_training(req.dataset_id, resume_from=req.resume_from)
    )


@router.post("/training/stop", response_model=TrainingStatus)
async def stop_training(studio=Depends(get_studio)) -> TrainingStatus:
    """Request a stop after the current step; returns immediately (state ``stopping``)."""
    return await run_in_threadpool(lambda: studio.stop_training(wait=False))


@router.post("/training/pause", response_model=TrainingStatus)
async def pause_training(studio=Depends(get_studio)) -> TrainingStatus:
    return await run_in_threadpool(studio.pause_training)


@router.post("/training/resume", response_model=TrainingStatus)
async def resume_training(studio=Depends(get_studio)) -> TrainingStatus:
    return await run_in_threadpool(studio.resume_training)


@router.get("/training/status", response_model=TrainingStatus)
async def training_status(studio=Depends(get_studio)) -> TrainingStatus:
    return studio.training_status()


@router.get("/training/metrics", response_model=list[MetricPoint])
async def training_metrics(
    since_step: int | None = Query(None, ge=0), studio=Depends(get_studio)
) -> list[MetricPoint]:
    return studio.training_metrics(since_step)


@router.get("/training/validate", response_model=TrainingValidation)
async def validate_training(
    dataset_id: str | None = Query(None, max_length=128), studio=Depends(get_studio)
) -> TrainingValidation:
    steps, warnings = await run_in_threadpool(studio.validate_training, dataset_id)
    return TrainingValidation(planned_steps=steps, warnings=list(warnings))


@router.get("/sessions", response_model=list[SessionSummary])
async def sessions(studio=Depends(get_studio)) -> list[SessionSummary]:
    return studio.sessions()
