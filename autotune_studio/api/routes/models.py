"""Base model configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...contracts import ModelConfig
from ...errors import NotFound, ValidationFailed
from ..deps import get_studio

router = APIRouter(tags=["model"])


@router.get("/model", response_model=ModelConfig)
async def get_model(studio=Depends(get_studio)) -> ModelConfig:
    cfg = studio.model_config
    if cfg is None:
        raise NotFound("no model configured yet")
    return cfg


@router.put("/model", response_model=ModelConfig)
async def put_model(cfg: ModelConfig, studio=Depends(get_studio)) -> ModelConfig:
    """Validate (repo id or local directory under an allowed root) and store the model config."""
    fields = cfg.model_dump(exclude={"model_name", "trust_remote_code"})
    try:
        return await run_in_threadpool(lambda: studio.configure_model(cfg.model_name, **fields))
    except ValidationFailed as exc:
        if exc.detail is None:  # let the UI attach the message to the model_name input
            exc.detail = {"field": "model_name"}
        raise


@router.get("/models/suggested", response_model=list[str])
async def suggested_models(studio=Depends(get_studio)) -> list[str]:
    return studio.suggested_models()
