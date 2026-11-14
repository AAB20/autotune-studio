"""Runtime, hardware, resources and launch info (authenticated)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from ...contracts import EnvironmentInfo, HardwareInfo, LaunchInfo, ResourceSnapshot
from ..deps import get_studio

router = APIRouter(tags=["system"])


@router.get("/env", response_model=EnvironmentInfo)
async def get_env(studio=Depends(get_studio)) -> EnvironmentInfo:
    return studio.detect_environment()


@router.get("/hardware", response_model=HardwareInfo)
async def get_hardware(refresh: bool = Query(False), studio=Depends(get_studio)) -> HardwareInfo:
    return await run_in_threadpool(studio.detect_hardware, refresh)


@router.get("/resources", response_model=ResourceSnapshot)
async def get_resources(studio=Depends(get_studio)) -> ResourceSnapshot:
    return await run_in_threadpool(studio.resources)


@router.get("/launch", response_model=LaunchInfo | None)
async def get_launch(studio=Depends(get_studio)) -> LaunchInfo | None:
    """Launch info with ``open_url`` stripped: it carries the session token in its fragment."""
    info = studio.launch_info
    if info is None:
        return None
    return info.model_copy(update={"open_url": None})
