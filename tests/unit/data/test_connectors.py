"""Connector-level tests: load, schema, empty inputs, malformed records, cardinality."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotune_studio.connectors import (
    CSVConnector,
    HuggingFaceConnector,
    JSONConnector,
    JSONLConnector,
    MalformedRecord,
    ParquetConnector,
    make_connector,
)
from autotune_studio.contracts import DatasetSpec
from autotune_studio.errors import DatasetEmpty, DatasetMalformed
from autotune_studio.interfaces import Connector, StreamCursor

from .helpers import SMALL, make_rows, write_csv, write_json_array, write_jsonl, write_parquet


def collect(conn: Connector, cursor: StreamCursor | None = None, sink=None) -> tuple[list[dict], list[StreamCursor]]:
    rows: list[dict] = []
    curs: list[StreamCursor] = []
    for batch, cur in conn.iter_batches(cursor, on_malformed=sink) if sink else conn.iter_batches(cursor):
        assert isinstance(batch, pa.RecordBatch)
        rows += batch.to_pylist()
        curs.append(cur)
    return rows, curs


@pytest.mark.parametrize("kind", ["csv", "jsonl", "json", "parquet"])
def test_make_connector_and_protocol(tmp_path: Path, kind: str) -> None:
    rows = make_rows(50)
    writer = {"csv": write_csv, "jsonl": write_jsonl, "json": write_json_array, "parquet": write_parquet}[kind]
    p = writer(tmp_path / f"d.{kind}", rows)
    conn = make_connector(DatasetSpec(source=kind, path=str(p)), SMALL)
    assert isinstance(conn, Connector)
    got, curs = collect(conn)
    assert len(got) == 50
    assert curs[-1].rows_consumed == 50
    assert conn.total_bytes() == p.stat().st_size
    if kind == "csv":  # CSV reads as strings by default
        assert got[3] == {k: str(v) for k, v in rows[3].items()}
    else:
        assert got[3] == rows[3]
    json.dumps(curs[-1].to_dict())  # cursors are JSON-serialisable


def test_make_connector_huggingface_lazy() -> None:
    conn = make_connector(DatasetSpec(source="huggingface", name="org/ds", split="test"), SMALL)
    assert isinstance(conn, HuggingFaceConnector) and conn.split == "test"


def test_multiple_batches_and_byte_budget(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "d.jsonl", make_rows(3000))
    rows, curs = collect(JSONLConnector(p, SMALL))
    assert len(rows) == 3000
    assert len(curs) > 3  # batched by bytes, not one giant batch
    assert [c.rows_consumed for c in curs] == sorted(c.rows_consumed for c in curs)
    assert curs[-1].state["offset"] == p.stat().st_size


def test_csv_empty_and_header_only(tmp_path: Path) -> None:
    empty = tmp_path / "e.csv"
    empty.write_bytes(b"")
    header = tmp_path / "h.csv"
    header.write_text("instruction,output\n")
    assert collect(CSVConnector(empty, SMALL))[0] == []
    assert collect(CSVConnector(header, SMALL))[0] == []


def test_csv_infer_types_and_string_default(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,x\n2,\n")
    assert collect(CSVConnector(p, SMALL))[0] == [{"a": "1", "b": "x"}, {"a": "2", "b": None}]
    assert collect(CSVConnector(p, SMALL, infer_types=True))[0][0]["a"] == 1


def test_jsonl_malformed_lines_reported_not_dropped(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(b'{"text": "a"}\n{not json\n\n[1, 2]\n\xff\xfe\n{"text": "b"}\n{"text": "c"}')
    seen: list[MalformedRecord] = []
    rows, _ = collect(JSONLConnector(p, SMALL), sink=seen.append)
    assert [r["text"] for r in rows] == ["a", "b", "c"]
    assert len(seen) == 3
    assert all(r.location.startswith("byte ") for r in seen)
    # without a sink the connector still counts them
    conn = JSONLConnector(p, SMALL)
    collect(conn)
    assert conn.malformed_total == 3


def test_jsonl_mixed_types_coerced_without_loss(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "x.jsonl", [{"a": 1, "text": "x"}, {"a": "str", "text": "y"}, {"text": "z"}])
    conn = JSONLConnector(p, SMALL)
    rows, _ = collect(conn)
    assert [r["text"] for r in rows] == ["x", "y", "z"]
    assert [r["a"] for r in rows] == ["1", "str", None]
    assert conn.coerced_batches == 1


def test_jsonl_line_longer_than_block(tmp_path: Path) -> None:
    big = "x" * (3 * SMALL)
    p = write_jsonl(tmp_path / "big.jsonl", [{"text": big}, {"text": "small"}])
    rows, _ = collect(JSONLConnector(p, SMALL))
    assert [len(r["text"]) for r in rows] == [len(big), 5]


def test_json_array_and_non_object_items(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text('[{"text": "a", "f": 1.5}, 3, {"text": "b", "f": 2}]')
    seen: list[MalformedRecord] = []
    rows, _ = collect(JSONConnector(p, SMALL), sink=seen.append)
    assert rows == [{"text": "a", "f": 1.5}, {"text": "b", "f": 2.0}]
    assert len(seen) == 1 and "item 1" in seen[0].location


def test_json_delegates_to_jsonl(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "lines.json", make_rows(10))
    conn = JSONConnector(p, SMALL)
    assert conn.kind == "jsonl"
    assert len(collect(conn)[0]) == 10


def test_json_records_key_and_single_object(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"version": 1, "data": make_rows(5)}, indent=2))
    conn = JSONConnector(p, SMALL)
    assert len(collect(conn)[0]) == 5 and "data" in (conn.layout_note or "")
    # a chat row (messages list) in a single object is one row, not a records array
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}, indent=2))
    assert collect(JSONConnector(s, SMALL))[0] == [{"messages": [{"role": "user", "content": "hi"}]}]


def test_json_syntax_error_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('[{"text": "a"}, {"text": ')
    with pytest.raises(DatasetMalformed):
        collect(JSONConnector(p, SMALL))
    q = tmp_path / "bad2.json"
    q.write_text("hello")
    with pytest.raises(DatasetMalformed):
        JSONConnector(q, SMALL)


def test_json_empty_file_and_empty_array(tmp_path: Path) -> None:
    e = tmp_path / "e.json"
    e.write_text("")
    a = tmp_path / "a.json"
    a.write_text("[]")
    assert collect(JSONConnector(e, SMALL))[0] == []
    assert collect(JSONConnector(a, SMALL))[0] == []


def test_cardinality_unknown_vs_exact(tmp_path: Path) -> None:
    rows = make_rows(1000)
    assert JSONLConnector(write_jsonl(tmp_path / "d.jsonl", rows), SMALL).cardinality().rows is None
    assert CSVConnector(write_csv(tmp_path / "d.csv", rows), SMALL).cardinality().rows is None
    assert JSONConnector(write_json_array(tmp_path / "d.json", rows), SMALL).cardinality().rows is None
    card = ParquetConnector(write_parquet(tmp_path / "d.parquet", rows), SMALL).cardinality()
    assert (card.rows, card.exact, card.method) == (1000, True, "metadata")


def test_parquet_directory_sorted(tmp_path: Path) -> None:
    d = tmp_path / "pq"
    (d / "sub").mkdir(parents=True)
    rows = make_rows(90)
    write_parquet(d / "b.parquet", rows[30:60])
    write_parquet(d / "a.parquet", rows[:30])
    write_parquet(d / "sub" / "c.parquet", rows[60:])
    conn = ParquetConnector(d, SMALL)
    assert conn.cardinality().rows == 90
    assert [r["id"] for r in collect(conn)[0]] == list(range(90))


def test_parquet_empty(tmp_path: Path) -> None:
    p = tmp_path / "e.parquet"
    pq.write_table(pa.table({"text": pa.array([], pa.string())}), p)
    conn = ParquetConnector(p, SMALL)
    assert conn.cardinality().rows == 0
    assert collect(conn)[0] == []
    (tmp_path / "none").mkdir()
    with pytest.raises(DatasetEmpty):
        ParquetConnector(tmp_path / "none", SMALL)


def test_cursor_kind_mismatch_rejected(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "d.jsonl", make_rows(3))
    with pytest.raises(ValueError):
        list(JSONLConnector(p, SMALL).iter_batches(StreamCursor(kind="csv", rows_consumed=1)))


def test_huggingface_local_json_streaming(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "hf.jsonl", make_rows(500))
    conn = HuggingFaceConnector("json", SMALL, data_files=str(p))
    rows, curs = collect(conn)
    assert len(rows) == 500 and rows[7] == make_rows(500)[7]
    assert len(curs) > 1
    assert "hf_state" in curs[0].state
    json.dumps(curs[0].to_dict())
    assert conn.cardinality().rows is None  # packaged json builder has no split metadata


def test_huggingface_metadata_cardinality_from_split_info() -> None:
    class _Split:
        num_examples, num_bytes = 1234, 99999

    class _Info:
        splits = {"train": _Split()}  # noqa: RUF012

    class _DS:
        info = _Info()

    conn = HuggingFaceConnector("org/ds", SMALL)
    conn._info_ds = _DS()
    card = conn.cardinality()
    assert (card.rows, card.exact, card.method) == (1234, True, "metadata")
    assert conn.total_bytes() == 99999
    conn.split = "validation"  # split absent from metadata -> unknown
    assert conn.cardinality().rows is None and conn.total_bytes() is None


def test_huggingface_never_passes_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    seen: dict = {}

    def fake_load(path, name=None, **kw):
        seen.update(kw, path=path, name=name)
        raise RuntimeError("stop")

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_test_token")
    conn = HuggingFaceConnector("org/ds", SMALL, config_name="cfg", revision="abc123", split="test")
    with pytest.raises(RuntimeError):
        list(conn.iter_batches())
    assert "trust_remote_code" not in seen
    assert seen["streaming"] is True and seen["split"] == "test" and seen["revision"] == "abc123"
    assert seen["name"] == "cfg" and seen["token"] == "hf_dummy_test_token"


def test_connectors_import_does_not_load_datasets() -> None:
    import subprocess
    import sys

    code = "import sys, autotune_studio.connectors, autotune_studio.data_engine; print('datasets' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.network
@pytest.mark.skipif(not __import__("os").environ.get("AUTOTUNE_RUN_NETWORK"),
                    reason="set AUTOTUNE_RUN_NETWORK=1 to run Hub tests")
def test_huggingface_hub_streaming_network() -> None:
    from autotune_studio.data_engine import open_dataset

    ds = open_dataset(DatasetSpec(source="huggingface", name="stanfordnlp/imdb", split="test"))
    assert ds.info.detected_format == "text"
    assert ds.info.cardinality.method in ("metadata", "unknown")
    it = ds.iter_rows()
    row, cur = next(it)
    it.close()
    assert "text" in row and cur.rows_consumed == 1
