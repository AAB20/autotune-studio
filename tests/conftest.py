"""Suite-wide pytest configuration and shared fixtures. Owned by: qa-testing.

Markers (declared in ``pyproject.toml``; re-registered here so the suite also works without it):

* ``network`` - needs internet. **Skipped** unless ``--run-network`` is passed or the environment
  variable ``AUTOTUNE_RUN_NETWORK`` is set to a truthy value (same variable already used by
  ``tests/unit/data``).
* ``slow``    - long-running (tiny-model training, 100+ MB files). Run by default; deselect with
  ``-m "not slow"``.
* ``gpu``     - needs CUDA. Skipped unless ``torch.cuda.is_available()``.

Shared fixtures:

* :func:`make_dataset` - synthetic dataset factory (csv / jsonl / json / parquet x
  instruction / chat / text, any size) written under ``tmp_path``.
* :func:`studio`       - a real :class:`~autotune_studio.studio.Studio` rooted in ``tmp_path``.
* :func:`auth_client`  - a FastAPI ``TestClient`` for that studio, authenticated with the Bearer token.

Sub-directory conftests (e.g. ``tests/unit/api``) may define their own ``studio``/``client``
fixtures; pytest's nearest-definition rule makes those win inside their packages.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import pytest

DatasetFmt = Literal["csv", "jsonl", "json", "parquet"]
DatasetKind = Literal["instruction", "chat", "text"]

_TRUTHY = {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- options / markers
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (or set AUTOTUNE_RUN_NETWORK=1)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running tests (tiny-model training, large files)")
    config.addinivalue_line("markers", "network: requires internet access (opt-in: --run-network)")
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


def network_enabled(config: pytest.Config) -> bool:
    """True when network tests were opted into via CLI flag or ``AUTOTUNE_RUN_NETWORK``."""
    if config.getoption("--run-network", default=False):
        return True
    return os.environ.get("AUTOTUNE_RUN_NETWORK", "").strip().lower() in _TRUTHY


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken -> no GPU tests
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = network_enabled(config)
    skip_network = pytest.mark.skip(reason="network test: pass --run-network or set AUTOTUNE_RUN_NETWORK=1")
    gpu_items = [it for it in items if "gpu" in it.keywords]
    has_cuda = _cuda_available() if gpu_items else False
    skip_gpu = pytest.mark.skip(reason="gpu test: CUDA is not available on this machine")
    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# --------------------------------------------------------------------------- dataset factories
def make_records(kind: DatasetKind, n: int, *, pad: int = 0) -> Iterator[dict[str, Any]]:
    """Yield ``n`` deterministic records of the given format. ``pad`` adds that many filler chars."""
    filler = ("lorem ipsum " * (pad // 12 + 1))[:pad] if pad else ""
    for i in range(n):
        if kind == "instruction":
            yield {"instruction": f"question {i}", "output": f"answer {i} {filler}".rstrip()}
        elif kind == "chat":
            yield {"messages": [{"role": "user", "content": f"hello {i}"},
                                {"role": "assistant", "content": f"reply {i} {filler}".rstrip()}]}
        elif kind == "text":
            yield {"text": f"sample text number {i} {filler}".rstrip()}
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown kind {kind!r}")


def write_dataset(path: Path, fmt: DatasetFmt, kind: DatasetKind = "instruction", n: int = 50,
                  *, pad: int = 0) -> Path:
    """Write ``n`` synthetic rows to ``path`` in ``fmt``; streams rows (no full materialisation for text formats)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = make_records(kind, n, pad=pad)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    elif fmt == "json":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("[")
            for i, r in enumerate(rows):
                f.write(("," if i else "") + json.dumps(r))
            f.write("]")
    elif fmt == "csv":
        if kind == "chat":
            raise ValueError("chat (nested messages) has no CSV representation")
        import csv

        with open(path, "w", encoding="utf-8", newline="") as f:
            w: csv.DictWriter | None = None
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)
    elif fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(list(rows)), path)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return path


@pytest.fixture
def make_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``make_dataset(fmt="jsonl", kind="instruction", n=50, name=None, pad=0) -> Path``."""

    def _make(fmt: DatasetFmt = "jsonl", kind: DatasetKind = "instruction", n: int = 50,
              name: str | None = None, pad: int = 0) -> Path:
        fname = name or f"{kind}-{n}.{fmt}"
        return write_dataset(tmp_path / "data" / fname, fmt, kind, n, pad=pad)

    return _make


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file (used to prove datasets are never modified)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- studio / client
@pytest.fixture
def studio(tmp_path: Path) -> Iterator[Any]:
    """A real Studio whose workspace and only extra allowed root is ``tmp_path``."""
    from autotune_studio.studio import Studio

    s = Studio(workspace_dir=tmp_path, allowed_roots=[str(tmp_path)], resource_interval=0.5)
    yield s
    try:
        if s.lifecycle.is_active:
            s.lifecycle.stop(wait=True, timeout=60)
    finally:
        s.shutdown(stop_training=False)


@pytest.fixture
def auth_client(studio: Any) -> Iterator[Any]:
    """``TestClient`` bound to :func:`studio`'s app with ``Authorization: Bearer <token>``."""
    from fastapi.testclient import TestClient

    with TestClient(studio.create_app(), raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {studio.auth.token}"
        yield c
