"""Fixtures for training tests: fully local tiny Llama model + tokenizer, synthetic datasets, fake sink."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

CHAT_TEMPLATE = (
    "{% for m in messages %}<s>{{ m['role'] }}: {{ m['content'] }}</s>{% endfor %}"
)


def build_tiny_model(dest: Path, chat_template: bool = False) -> Path:
    """Write a ~100 KB Llama (hidden 16, 1 layer) + byte-level BPE tokenizer to ``dest`` (no network)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    dest.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [f"### Instruction:\nquestion {i}\n\n### Response:\nanswer {i} user assistant" for i in range(300)]
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   pad_token="<pad>")
    if chat_template:
        fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(dest)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                      bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                      pad_token_id=fast.pad_token_id, tie_word_embeddings=False)
    import torch

    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(dest)
    return dest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny_llama"))


def make_rows(n: int) -> list[dict[str, Any]]:
    return [{"instruction": f"question {i}", "output": f"answer {i}"} for i in range(n)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def open_jsonl(path: Path):
    from autotune_studio.contracts import DatasetSource, DatasetSpec
    from autotune_studio.data_engine import open_dataset

    return open_dataset(DatasetSpec(source=DatasetSource.JSONL, path=str(path)))


@pytest.fixture
def jsonl_dataset(tmp_path: Path):
    return open_jsonl(write_jsonl(tmp_path / "train.jsonl", make_rows(40)))


class FakeSink:
    """Records events; optional ``hook(type, data)`` runs on the publishing (training) thread."""

    def __init__(self, hook: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.hook = hook
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def publish(self, type: str, data: dict[str, Any]) -> None:
        with self.cond:
            self.events.append((type, dict(data)))
            self.cond.notify_all()
        if self.hook is not None:
            self.hook(type, data)

    def of(self, type: str) -> list[dict[str, Any]]:
        with self.lock:
            return [d for t, d in self.events if t == type]

    def wait_for(self, pred: Callable[[str, dict[str, Any]], bool], timeout: float = 60.0) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: any(pred(t, d) for t, d in self.events), timeout)
