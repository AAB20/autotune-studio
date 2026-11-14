"""Schema and format discovery from a byte-bounded sample (never a row count)."""

from __future__ import annotations

import uuid
from typing import Any

import pyarrow as pa

from ..connectors.base import MalformedRecord, iter_with_sink
from ..contracts import ColumnInfo, DatasetFormat, DatasetInfo, DatasetSpec
from ..errors import ValidationFailed
from ..interfaces import Connector
from .memory_manager import MiB
from .validation import RowValidator, canonical_json, ensure_usable

DEFAULT_SAMPLE_BYTES = 4 * MiB
EXAMPLE_CHARS = 120

CHAT_COLUMNS = ("messages", "conversations")
TEXT_COLUMNS = ("text", "content", "document")
INSTRUCTION_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (instruction column, optional input columns, output columns)
    ("instruction", ("input", "context"), ("output", "response", "answer")),
    ("prompt", (), ("response", "completion", "answer", "output")),
    ("question", (), ("answer", "response")),
)


def new_dataset_id() -> str:
    """Short random hex id."""
    return uuid.uuid4().hex[:12]


def _is_chat_value(v: Any) -> bool:
    if not isinstance(v, list) or not v:
        return False
    return all(isinstance(m, dict) and ("role" in m or "from" in m) and ("content" in m or "value" in m) for m in v)


def detect_format(rows: list[dict[str, Any]], schema: pa.Schema | None,
                  text_column: str | None = None) -> tuple[DatasetFormat, list[str], list[str]]:
    """Return ``(format, text_columns, warnings)`` for sampled ``rows``."""
    warnings: list[str] = []
    names: list[str] = list(schema.names) if schema is not None else []
    for r in rows:
        for k in r:
            if k not in names:
                names.append(k)
    if text_column is not None:
        if text_column not in names:
            raise ValidationFailed(f"text_column {text_column!r} not found; columns: {names}",
                                   {"text_column": text_column, "columns": names})
        return "text", [text_column], warnings
    lower = {n.lower(): n for n in names}

    for c in CHAT_COLUMNS:
        col = lower.get(c)
        if col is None:
            continue
        values = [r.get(col) for r in rows if r.get(col) is not None]
        if values and sum(_is_chat_value(v) for v in values) * 2 >= len(values):
            return "chat", [col], warnings

    for instr, inputs, outputs in INSTRUCTION_PATTERNS:
        if instr not in lower:
            continue
        out = next((lower[o] for o in outputs if o in lower), None)
        if out is None:
            continue
        inp = next((lower[i] for i in inputs if i in lower), None)
        cols = [lower[instr]] + ([inp] if inp else []) + [out]
        return "instruction", cols, warnings

    def is_string_col(n: str) -> bool:
        if schema is not None and n in schema.names:
            t = schema.field(n).type
            return pa.types.is_string(t) or pa.types.is_large_string(t)
        vals = [r.get(n) for r in rows if r.get(n) is not None]
        return bool(vals) and all(isinstance(v, str) for v in vals)

    for c in TEXT_COLUMNS:
        col = lower.get(c)
        if col is not None and is_string_col(col):
            return "text", [col], warnings
    str_cols = [n for n in names if is_string_col(n)]
    if len(str_cols) == 1:
        return "text", str_cols, warnings
    if str_cols:
        def avg_len(n: str) -> float:
            vals = [len(r[n]) for r in rows if isinstance(r.get(n), str)]
            return sum(vals) / len(vals) if vals else 0.0
        best = max(str_cols, key=avg_len)
        warnings.append(f"no standard text column; using longest string column {best!r} "
                        f"(of {str_cols}); set text_column to override")
        return "text", [best], warnings
    warnings.append("could not detect a chat/instruction/text layout; rows will be rendered as JSON. "
                    "Set text_column to choose the training text.")
    return "unknown", [], warnings


def _unify(schemas: list[pa.Schema], warnings: list[str]) -> pa.Schema | None:
    if not schemas:
        return None
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        warnings.append("column types differ between sampled batches; showing types of the first batch")
        return schemas[0]


def _example(v: Any) -> str:
    s = v if isinstance(v, str) else canonical_json(v)
    return s if len(s) <= EXAMPLE_CHARS else s[: EXAMPLE_CHARS - 3] + "..."


def discover(connector: Connector, spec: DatasetSpec, sample_bytes: int = DEFAULT_SAMPLE_BYTES,
             dataset_id: str | None = None) -> DatasetInfo:
    """Sample up to ``sample_bytes`` (canonical-JSON bytes) of rows and build a :class:`DatasetInfo`.

    Raises :class:`DatasetEmpty` when the source yields no record at all and
    :class:`DatasetMalformed` when every sampled record is malformed.
    """
    malformed: list[MalformedRecord] = []
    rows: list[dict[str, Any]] = []
    schemas: list[pa.Schema] = []
    sampled_bytes = 0
    gen = iter_with_sink(connector, None, malformed.append)
    try:
        for batch, _cur in gen:
            schemas.append(batch.schema)
            for row in batch.to_pylist():
                rows.append(row)
                sampled_bytes += len(canonical_json(row).encode("utf-8")) + 1
                if sampled_bytes >= sample_bytes:
                    break
            if sampled_bytes >= sample_bytes:
                break
    finally:
        gen.close()  # type: ignore[attr-defined]

    source = spec.name or spec.path or "?"
    if not rows:
        ensure_usable(0, len(malformed), [m.describe() for m in malformed[:5]], source)

    warnings: list[str] = []
    schema = _unify(schemas, warnings)
    fmt, text_cols, fmt_warnings = detect_format(rows, schema, spec.text_column)
    warnings += fmt_warnings

    info = DatasetInfo(
        dataset_id=dataset_id or new_dataset_id(),
        spec=spec,
        columns=[],
        cardinality=connector.cardinality(),
        detected_format=fmt,
        text_columns=text_cols,
        sampled_rows=len(rows),
        sampled_bytes=sampled_bytes,
    )
    validator = RowValidator(info)
    for m in malformed:
        validator.record_malformed(m.describe())
    for i, row in enumerate(rows):
        validator.check(row, f"sample row {i}")
    ensure_usable(validator.accepted_rows, validator.malformed_rows, validator.malformed_examples, source)
    if validator.malformed_rows:
        warnings.append(f"{validator.malformed_rows} malformed record(s) in the sample of "
                        f"{len(rows) + len(malformed)}; they are skipped and counted during streaming")

    names: list[str] = list(schema.names) if schema is not None else []
    columns: list[ColumnInfo] = []
    for n in names:
        values = [r.get(n) for r in rows]
        nulls = sum(v is None for v in values)
        examples = [_example(v) for v in values if v is not None][:3]
        columns.append(ColumnInfo(name=n, dtype=str(schema.field(n).type) if schema is not None else "unknown",
                                  nullable=True, sampled_null_count=nulls, examples=examples))
    if info.cardinality.rows is None:
        warnings.append("row count unknown (streaming source); training needs max_steps, or run a scan")
    return info.model_copy(update={"columns": columns, "warnings": warnings})
