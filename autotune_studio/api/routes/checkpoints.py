"""Checkpoints, resume and model export."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import Path as PathParam
from starlette.concurrency import run_in_threadpool

from ...contracts import CheckpointInfo, ExportRequest, ExportResult, TrainingStatus
from ..deps import get_studio
from ..schemas import ResumeCheckpointRequest

router = APIRouter(tags=["checkpoints"])

#: Kept below Cloudflare's ~100 s origin response timeout so the client gets a JSON error
#: (409 conflict) rather than a proxy 524 if the engine is between long steps.
SAVE_CHECKPOINT_TIMEOUT = 90.0

CheckpointId = PathParam(..., min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._-]+$")


@router.get("/checkpoints", response_model=list[CheckpointInfo])
async def list_checkpoints(studio=Depends(get_studio)) -> list[CheckpointInfo]:
    return await run_in_threadpool(studio.list_checkpoints)


@router.post("/checkpoints", response_model=CheckpointInfo)
async def save_checkpoint(studio=Depends(get_studio)) -> CheckpointInfo:
    """Save a checkpoint of the running job at the next step boundary (blocks until written)."""
    return await run_in_threadpool(lambda: studio.save_checkpoint(timeout=SAVE_CHECKPOINT_TIMEOUT))


@router.post("/checkpoints/{checkpoint_id}/resume", response_model=TrainingStatus)
async def resume_checkpoint(
    checkpoint_id: str = CheckpointId,
    body: ResumeCheckpointRequest | None = None,
    studio=Depends(get_studio),
) -> TrainingStatus:
    req = body or ResumeCheckpointRequest()
    return await run_in_threadpool(lambda: studio.resume_from_checkpoint(checkpoint_id, req.dataset_id))


@router.post("/export", response_model=ExportResult)
async def export_model(req: ExportRequest, studio=Depends(get_studio)) -> ExportResult:
    """Export the LoRA adapter or a merged 16-bit model (``output_dir`` must be inside the allowed roots)."""
    return await run_in_threadpool(
        lambda: studio.export_model(
            req.checkpoint_id, format=req.format, output_dir=req.output_dir,
            push_to_hub=req.push_to_hub, hub_repo_name=req.hub_repo_name, hub_private=req.hub_private,
        )
    )
