"""Dataset connectors (CSV, JSON, JSONL, Parquet, Hugging Face).

Each connector implements :class:`autotune_studio.interfaces.Connector` and
additionally accepts ``on_malformed`` in ``iter_batches`` (see
:mod:`autotune_studio.connectors.base`). Path policy (allowed roots, symlinks)
is enforced by the caller through the security module before
:func:`make_connector` is called.
"""

from __future__ import annotations

from ..contracts import DatasetSource, DatasetSpec
from ..interfaces import Connector
from .base import MalformedRecord, MalformedSink, rows_to_batch
from .csv import CSVConnector
from .huggingface import HuggingFaceConnector
from .json import JSONConnector
from .jsonl import JSONLConnector
from .parquet import ParquetConnector

__all__ = [
    "CSVConnector",
    "HuggingFaceConnector",
    "JSONConnector",
    "JSONLConnector",
    "MalformedRecord",
    "MalformedSink",
    "ParquetConnector",
    "make_connector",
    "rows_to_batch",
]


def make_connector(spec: DatasetSpec, batch_bytes: int) -> Connector:
    """Build the connector for ``spec`` with batches of roughly ``batch_bytes``.

    Assumes ``spec.path`` was already validated by the security module.
    """
    src = DatasetSource(spec.source)
    if src == DatasetSource.HUGGINGFACE:
        assert spec.name  # guaranteed by DatasetSpec validation
        return HuggingFaceConnector(spec.name, batch_bytes, split=spec.split, config_name=spec.config_name,
                                    revision=spec.revision)
    assert spec.path
    if src == DatasetSource.CSV:
        return CSVConnector(spec.path, batch_bytes)
    if src == DatasetSource.JSONL:
        return JSONLConnector(spec.path, batch_bytes)
    if src == DatasetSource.JSON:
        return JSONConnector(spec.path, batch_bytes)
    if src == DatasetSource.PARQUET:
        return ParquetConnector(spec.path, batch_bytes)
    raise ValueError(f"unsupported dataset source: {spec.source!r}")  # pragma: no cover
