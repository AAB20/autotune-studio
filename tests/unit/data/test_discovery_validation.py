"""Discovery (schema, format detection) and validation (malformed rows, checksum, to_text)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotune_studio.contracts import DatasetSpec
from autotune_studio.data_engine import RowValidator, canonical_json, open_dataset, rows_checksum, to_text
from autotune_studio.errors import DatasetEmpty, DatasetMalformed, ValidationFailed

from .helpers import make_rows, small_budget, write_csv, write_json_array, write_jsonl, write_parquet


def _open(tmp_path: Path, rows: list[dict], kind: str = "jsonl", **spec_kw):
    writer = {"csv": write_csv, "jsonl": write_jsonl, "json": write_json_array, "parquet": write_parquet}[kind]
    p = writer(tmp_path / f"d.{kind}", rows)
    return open_dataset(DatasetSpec(source=kind, path=str(p), **spec_kw), budget=small_budget())


@pytest.mark.parametrize("kind", ["csv", "jsonl", "json", "parquet"])
def test_schema_and_instruction_detection_all_formats(tmp_path: Path, kind: str) -> None:
    ds = _open(tmp_path, make_rows(20), kind)
    info = ds.info
    assert [c.name for c in info.columns] == ["id", "instruction", "output"]
    assert info.detected_format == "instruction"
    assert info.text_columns == ["instruction", "output"]
    assert info.sampled_rows == 20 and info.sampled_bytes > 0
    assert len(info.dataset_id) == 12
    col = info.columns[1]
    assert col.dtype == "string" and len(col.examples) == 3 and all(len(e) <= 120 for e in col.examples)
    if kind == "parquet":
        assert info.cardinality.rows == 20 and info.cardinality.exact
    else:
        assert info.cardinality.rows is None
        assert any("max_steps" in w for w in info.warnings)


def test_chat_detection_and_to_text(tmp_path: Path) -> None:
    rows = [{"messages": [{"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": "hello"}]}
            for i in range(5)]
    ds = _open(tmp_path, rows)
    assert ds.info.detected_format == "chat" and ds.info.text_columns == ["messages"]
    assert ds.to_text(rows[0]) == "<|user|>\nhi 0\n<|assistant|>\nhello"


def test_sharegpt_conversations_detection(tmp_path: Path) -> None:
    rows = [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}] * 3
    ds = _open(tmp_path, rows)
    assert ds.info.detected_format == "chat"
    assert ds.to_text(rows[0]) == "<|user|>\nq\n<|assistant|>\na"


def test_instruction_with_input_alpaca(tmp_path: Path) -> None:
    rows = [{"instruction": "Add", "input": "1+1", "output": "2"}, {"instruction": "Say hi", "input": "", "output": "hi"}]
    ds = _open(tmp_path, rows)
    assert ds.info.text_columns == ["instruction", "input", "output"]
    assert ds.to_text(rows[0]) == "### Instruction:\nAdd\n\n### Input:\n1+1\n\n### Response:\n2"
    assert "### Input" not in ds.to_text(rows[1])


def test_prompt_response_and_question_answer(tmp_path: Path) -> None:
    ds = _open(tmp_path, [{"prompt": "p", "completion": "c"}])
    assert ds.info.detected_format == "instruction" and ds.info.text_columns == ["prompt", "completion"]
    (tmp_path / "q").mkdir()
    ds2 = _open(tmp_path / "q", [{"question": "q", "answer": "a"}])
    assert ds2.info.text_columns == ["question", "answer"]


def test_text_detection(tmp_path: Path) -> None:
    ds = _open(tmp_path, [{"text": "hello world", "meta": 1}] * 3)
    assert ds.info.detected_format == "text" and ds.info.text_columns == ["text"]
    assert ds.to_text({"text": "hello world"}) == "hello world"


def test_text_detection_longest_string_column(tmp_path: Path) -> None:
    ds = _open(tmp_path, [{"title": "short", "body": "a much longer body of text", "n": 1}] * 3)
    assert ds.info.text_columns == ["body"]
    assert any("longest string column" in w for w in ds.info.warnings)


def test_unknown_format(tmp_path: Path) -> None:
    ds = _open(tmp_path, [{"a": 1, "b": 2.5}] * 3)
    assert ds.info.detected_format == "unknown" and ds.info.text_columns == []
    assert any("text_column" in w for w in ds.info.warnings)
    assert ds.to_text({"a": 1, "b": 2.5}) == canonical_json({"a": 1, "b": 2.5})


def test_text_column_override(tmp_path: Path) -> None:
    ds = _open(tmp_path, [{"a": "x", "text": "y"}] * 3, text_column="a")
    assert ds.info.detected_format == "text" and ds.info.text_columns == ["a"]
    with pytest.raises(ValidationFailed):
        _open(tmp_path, [{"a": "x"}], text_column="missing")


@pytest.mark.parametrize("kind,content", [("jsonl", b""), ("jsonl", b"\n\n  \n"), ("csv", b""),
                                          ("csv", b"instruction,output\n"), ("json", b"[]"), ("json", b"")])
def test_empty_sources_raise_dataset_empty(tmp_path: Path, kind: str, content: bytes) -> None:
    p = tmp_path / f"e.{kind}"
    p.write_bytes(content)
    with pytest.raises(DatasetEmpty):
        open_dataset(DatasetSpec(source=kind, path=str(p)), budget=small_budget())


def test_all_malformed_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{oops\nnot json\n[1]\n")
    with pytest.raises(DatasetMalformed) as ei:
        open_dataset(DatasetSpec(source="jsonl", path=str(p)), budget=small_budget())
    assert ei.value.detail and ei.value.detail["malformed_rows"] == 3


def test_all_rows_invalid_for_format_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetMalformed):
        _open(tmp_path, [{"text": ""}, {"text": "   "}, {"text": None}])


def test_partially_malformed_sample_warns_and_counts(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    good = [json.dumps({"text": f"t{i}"}) for i in range(8)]
    p.write_text("\n".join(good[:4] + ["{broken", '{"text": ""}'] + good[4:]) + "\n")
    ds = open_dataset(DatasetSpec(source="jsonl", path=str(p)), budget=small_budget())
    assert any("malformed" in w for w in ds.info.warnings)
    seen = [r["text"] for r, _ in ds.iter_rows()]
    assert seen == [f"t{i}" for i in range(8)]
    st = ds.stats
    assert st.malformed_rows == 2 and len(st.malformed_examples) == 2 and st.rows_seen == 8


def test_validator_keeps_at_most_five_examples_and_checksum() -> None:
    from autotune_studio.contracts import Cardinality, DatasetInfo

    info = DatasetInfo(dataset_id="x", spec=DatasetSpec(source="jsonl", path="p"), columns=[],
                       cardinality=Cardinality(), detected_format="text", text_columns=["text"])
    v = RowValidator(info)
    rows = [{"text": f"r{i}"} for i in range(5)]
    for i in range(9):
        assert not v.check({"text": ""}, f"row {i}")
    for r in rows:
        assert v.check(r)
    assert v.malformed_rows == 9 and len(v.malformed_examples) == 5
    assert v.accepted_rows == 5
    assert v.checksum == rows_checksum(rows) == rows_checksum(iter(rows))
    assert v.checksum != rows_checksum(rows[::-1])
    assert to_text({"text": 5}, info) == "5"
