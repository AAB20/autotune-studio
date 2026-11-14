"""Dataset loading, upload, preview and streaming statistics."""

from __future__ import annotations

import gc
import logging
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi import Path as PathParam
from starlette.concurrency import run_in_threadpool

from ...contracts import DatasetInfo, DatasetPreview, DatasetSource, DatasetSpec, StreamStats
from ...errors import DatasetMalformed, StudioError, ValidationFailed
from ...security import UploadLimits, asave_upload_stream, sanitize_upload_filename
from ..deps import get_settings, get_studio
from ..multipart import MultipartReader, parse_boundary

log = logging.getLogger("autotune_studio.api")

router = APIRouter(tags=["datasets"])

MAX_PREVIEW_ROWS = 1000

#: Upload extension -> dataset source.
EXT_SOURCES: dict[str, DatasetSource] = {
    ".csv": DatasetSource.CSV,
    ".tsv": DatasetSource.CSV,
    ".json": DatasetSource.JSON,
    ".jsonl": DatasetSource.JSONL,
    ".ndjson": DatasetSource.JSONL,
    ".parquet": DatasetSource.PARQUET,
    ".pq": DatasetSource.PARQUET,
}

DatasetId = PathParam(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")

# Allow multipart framing overhead on top of the file size for the early Content-Length check.
_MULTIPART_OVERHEAD = 64 * 1024


@router.post("/datasets", response_model=DatasetInfo)
async def create_dataset(spec: DatasetSpec, studio=Depends(get_studio)) -> DatasetInfo:
    """Open a dataset (local path inside the allowed roots, or a Hugging Face id) for streaming."""
    return await run_in_threadpool(studio.load_dataset_spec, spec)


@router.post("/datasets/upload", response_model=DatasetInfo)
async def upload_dataset(request: Request, studio=Depends(get_studio), settings=Depends(get_settings)) -> DatasetInfo:
    """Multipart upload (field ``file``) streamed to ``<workspace>/uploads/<random>.<ext>`` then loaded.

    The source is inferred from the extension (csv/tsv, json, jsonl/ndjson, parquet/pq). The
    stored file is deleted again if it cannot be opened as a dataset.
    """
    limits = UploadLimits(max_bytes=settings.upload_max_bytes)
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > limits.max_bytes + _MULTIPART_OVERHEAD:
        raise ValidationFailed("upload exceeds maximum size", {"max_bytes": limits.max_bytes})
    reader = MultipartReader(request.stream(), parse_boundary(request.headers.get("content-type")))
    dest: Path | None = None
    while (part := await reader.next_part()) is not None:
        if part.name != "file":
            continue  # unknown fields are drained, never buffered
        if not part.filename:
            raise ValidationFailed("the 'file' field must be a file upload", {"field": "file"})
        _, ext = sanitize_upload_filename(part.filename)
        dest = await asave_upload_stream(reader.iter_data(), studio.workspace / "uploads", ext, limits)
        break
    if dest is None:
        raise ValidationFailed("missing multipart field 'file'", {"field": "file"})
    spec = DatasetSpec(source=EXT_SOURCES[dest.suffix.lower()], path=str(dest))
    return await run_in_threadpool(_load_or_discard, studio, spec, dest)


def _load_or_discard(studio, spec: DatasetSpec, dest: Path) -> DatasetInfo:
    """Load an uploaded file; on failure delete it and raise a client error (runs in the threadpool).

    Parser exceptions on user-supplied bytes become ``dataset_malformed`` (422). The original
    exception is not kept (not chained, not logged as an object) and the failed load's frames are
    cleared *before* deleting: a reference to a half-constructed reader keeps the file open and
    Windows cannot delete open files (observed with pyarrow ``ParquetFile``).
    """
    error: BaseException
    try:
        return studio.load_dataset_spec(spec)
    except StudioError as exc:
        error = exc
    except Exception as exc:  # noqa: BLE001 - client-supplied file
        # No exc_info: a retained traceback (e.g. by a log handler) would keep the file open.
        log.warning("uploaded %s file could not be opened: %s: %s", spec.source.value, type(exc).__name__, str(exc))
        error = DatasetMalformed(
            f"the uploaded file could not be read as {spec.source.value}: {type(exc).__name__}"
        )
        error.__cause__ = None
    except BaseException:
        _remove_upload(dest)
        raise
    seen: set[int] = set()
    e: BaseException | None = error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if e.__traceback__ is not None:
            traceback.clear_frames(e.__traceback__)
        e = e.__cause__ or e.__context__
    error.__context__ = None
    gc.collect()
    _remove_upload(dest)
    raise error


def _remove_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not delete rejected upload %s (file still in use)", path.name)


@router.get("/datasets", response_model=list[DatasetInfo])
async def list_datasets(studio=Depends(get_studio)) -> list[DatasetInfo]:
    return studio.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str = DatasetId, studio=Depends(get_studio)) -> DatasetInfo:
    return studio.get_dataset(dataset_id).info


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: str = DatasetId,
    n: int = Query(20, ge=0, le=MAX_PREVIEW_ROWS),
    studio=Depends(get_studio),
) -> DatasetPreview:
    ds = studio.get_dataset(dataset_id)
    return await run_in_threadpool(ds.preview, n)


@router.get("/datasets/{dataset_id}/stats", response_model=StreamStats)
async def dataset_stats(dataset_id: str = DatasetId, studio=Depends(get_studio)) -> StreamStats:
    return studio.dataset_stats(dataset_id)


@router.post("/datasets/{dataset_id}/scan", response_model=StreamStats)
async def scan_dataset(dataset_id: str = DatasetId, studio=Depends(get_studio)) -> StreamStats:
    """Start a full streaming pass in the background (progress arrives as ``dataset`` events)."""
    return await run_in_threadpool(lambda: studio.scan_dataset(dataset_id, background=True))


@router.post("/datasets/{dataset_id}/cancel", response_model=StreamStats)
async def cancel_dataset(dataset_id: str = DatasetId, studio=Depends(get_studio)) -> StreamStats:
    return await run_in_threadpool(studio.cancel_dataset, dataset_id)
