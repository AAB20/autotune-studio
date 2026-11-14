"""Hugging Face credentials for pushing exports (ADR-0007). The token is write-only: never returned."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...contracts import HFCredentialsRequest, HFCredentialsStatus
from ..deps import get_studio

router = APIRouter(tags=["huggingface"])


@router.get("/hf/credentials", response_model=HFCredentialsStatus)
async def get_hf_credentials(studio=Depends(get_studio)) -> HFCredentialsStatus:
    return studio.hf_credentials_status()


@router.put("/hf/credentials", response_model=HFCredentialsStatus)
async def put_hf_credentials(body: HFCredentialsRequest, studio=Depends(get_studio)) -> HFCredentialsStatus:
    """Verify with the Hub (network call, run off the event loop) and keep in memory."""
    return await run_in_threadpool(lambda: studio.set_hf_credentials(body.username, body.token))


@router.delete("/hf/credentials", response_model=HFCredentialsStatus)
async def delete_hf_credentials(studio=Depends(get_studio)) -> HFCredentialsStatus:
    return studio.clear_hf_credentials()
