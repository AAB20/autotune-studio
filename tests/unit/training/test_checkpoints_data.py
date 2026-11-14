"""Checkpoint metadata, data positions, streaming encoder and collator (fast, no training)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from autotune_studio.contracts import ModelConfig, TrainingConfig
from autotune_studio.errors import DatasetEmpty, NotFound, ValidationFailed
from autotune_studio.interfaces import StreamCursor
from autotune_studio.training import checkpoints as ck
from autotune_studio.training.data import (
    SEQ_KEY,
    DataPosition,
    PositionRegistry,
    TrainStream,
    make_collator,
    make_encoder,
)

from .conftest import make_rows, open_jsonl, write_jsonl

M = ModelConfig(model_name="tiny")
T = TrainingConfig(method="lora", max_steps=3)


def test_write_load_list_checkpoints(tmp_path):
    root = tmp_path / "runs"
    a = ck.write_checkpoint_state(root / "job-a" / "checkpoint-3", job_id="job-a", step=3,
                                  data_cursor={"kind": "jsonl", "rows_consumed": 6, "state": {}, "pass": 0},
                                  model_cfg=M, train_cfg=T, is_final=False, backend="hf")
    (root / "job-a" / "checkpoint-3" / "adapter_model.safetensors").write_bytes(b"x" * 100)
    time.sleep(0.01)
    b = ck.write_checkpoint_state(root / "job-b" / "checkpoint-1", job_id="job-b", step=1, data_cursor=None,
                                  model_cfg=M, train_cfg=T, is_final=True)
    (root / "job-c" / "checkpoint-9").mkdir(parents=True)  # no state file -> ignored
    bad = root / "job-d" / "checkpoint-2"
    bad.mkdir(parents=True)
    (bad / ck.STATE_FILE).write_text("{not json", encoding="utf-8")  # corrupt -> skipped
    infos = ck.list_checkpoints(root)
    assert [i.checkpoint_id for i in infos] == [a.checkpoint_id, b.checkpoint_id]
    assert a.checkpoint_id == "job-a-checkpoint-3"
    info = ck.load_checkpoint_info(root / "job-a" / "checkpoint-3")
    assert info.size_bytes > 100 and info.data_cursor["rows_consumed"] == 6 and not info.is_final
    state = ck.load_checkpoint_state(info.path)
    assert state["model_config"]["model_name"] == "tiny" and state["training_config"]["max_steps"] == 3
    assert state["job_id"] == "job-a" and state["backend"] == "hf"
    assert ck.latest_checkpoint(root / "job-a").step == 3
    assert ck.list_checkpoints(tmp_path / "missing") == []
    with pytest.raises(NotFound):
        ck.load_checkpoint_state(root / "job-c" / "checkpoint-9")
    with pytest.raises(ValidationFailed):
        ck.load_checkpoint_state(bad)


def test_data_position_roundtrip():
    assert DataPosition().to_dict() is None
    p = DataPosition(2, StreamCursor(kind="jsonl", rows_consumed=5, state={"offset": 10}))
    d = p.to_dict()
    assert d["pass"] == 2 and StreamCursor.from_dict(d).rows_consumed == 5  # still a valid StreamCursor dict
    assert DataPosition.from_dict(json.loads(json.dumps(d))) == p
    assert DataPosition.from_dict(DataPosition(1, None).to_dict()) == DataPosition(1, None)


class _Tok:
    """Minimal tokenizer stub: one id per whitespace word."""

    eos_token_id = 2
    chat_template = None

    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=True):
        ids = ([1] if add_special_tokens else []) + [10 + len(w) for w in text.split()]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def test_train_stream_multi_pass_and_exact_resume(tmp_path):
    ds = open_jsonl(write_jsonl(tmp_path / "d.jsonl", make_rows(5)))
    enc = make_encoder(ds, _Tok(), 8)
    reg = PositionRegistry()
    gen = TrainStream(ds, enc, reg, DataPosition(), 0.0, 0)()
    items = [next(gen) for _ in range(12)]  # 5 rows/pass -> crosses two pass boundaries
    assert all(len(i["input_ids"]) <= 8 and i["input_ids"][-1] == 2 for i in items)
    reg.consume(items[6][SEQ_KEY], 7)
    pos = reg.last_consumed
    assert pos.pass_index == 1 and pos.cursor.rows_consumed == 2 and reg.rows_consumed == 7
    # resume from the recorded position reproduces exactly the rows after it
    reg2 = PositionRegistry(pos)
    gen2 = TrainStream(ds, enc, reg2, DataPosition.from_dict(pos.to_dict()), 0.0, 0)()
    resumed = [next(gen2)["input_ids"] for _ in range(5)]
    assert resumed == [i["input_ids"] for i in items[7:12]]


def test_train_stream_stops_on_event_and_rejects_empty_split(tmp_path):
    ds = open_jsonl(write_jsonl(tmp_path / "d.jsonl", make_rows(3)))
    stop = threading.Event()
    gen = TrainStream(ds, make_encoder(ds, _Tok(), 8), PositionRegistry(), DataPosition(), 0.0, 0, stop)()
    next(gen)
    stop.set()
    assert list(gen) == []
    # val_fraction close to 1 on 3 rows -> almost certainly empty train split; use 0.5 boundary rows check
    gen = TrainStream(ds, make_encoder(ds, _Tok(), 8), PositionRegistry(), DataPosition(), 0.999999, 0)()
    with pytest.raises(DatasetEmpty):
        list(gen)


def test_collator_masks_padding_and_forwards_seq():
    collate = make_collator(pad_token_id=0)
    b = collate([{"input_ids": [5, 6, 7], SEQ_KEY: 3}, {"input_ids": [8], SEQ_KEY: 4}])
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"].tolist() == [[5, 6, 7], [8, -100, -100]]
    assert b[SEQ_KEY] == 4 and b["_atk_rows"] == 2


def test_chat_rows_use_tokenizer_chat_template(tmp_path, tiny_model_dir):
    from transformers import AutoTokenizer

    from .conftest import CHAT_TEMPLATE

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    tok.chat_template = CHAT_TEMPLATE
    rows = [{"messages": [{"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": f"yo {i}"}]}
            for i in range(5)]
    ds = open_jsonl(write_jsonl(tmp_path / "chat.jsonl", rows))
    assert ds.info.detected_format == "chat"
    ids = make_encoder(ds, tok, 64)(rows[0])
    expected = tok("<s>user: hi 0</s><s>assistant: yo 0</s>", add_special_tokens=False)["input_ids"]
    assert ids == expected
    tok.chat_template = None  # fallback: data-engine text + EOS
    ids = make_encoder(ds, tok, 64)(rows[0])
    assert ids[-1] == tok.eos_token_id and "<|user|>" in tok.decode(ids)
