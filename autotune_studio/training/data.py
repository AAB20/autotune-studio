"""Streaming training data: rows -> text -> token ids, with exact data-cursor tracking.

How cursor tracking works (precision)
-------------------------------------
* Every training example yielded to the Trainer carries an integer ``_atk_seq`` that maps
  (in :class:`PositionRegistry`) to the :class:`DataPosition` *after* that row: the pass
  (epoch) index plus the data-engine :class:`StreamCursor` returned by ``iter_rows``.
* The collator forwards the highest ``_atk_seq`` of each micro-batch; the engine's
  ``Trainer.training_step`` override pops it and calls :meth:`PositionRegistry.consume`
  **only when the micro-batch is actually trained**. The Trainer may prefetch batches
  (it collects ``gradient_accumulation_steps`` micro-batches per optimizer step), but
  prefetched-but-untrained rows never advance the recorded position.
* A checkpoint at optimizer step N therefore stores the position right after the last row
  of the last micro-batch of step N: resuming from it replays no trained row and skips no
  untrained row (row-exact, given ``dataloader_num_workers=0`` and a single process, which
  the engine enforces). Rows from the validation split are skipped by the cursor naturally.
* Multiple epochs over a stream re-iterate the source (``pass`` += 1, cursor reset).
* Rows are never dropped: texts longer than ``max_seq_length`` are truncated (tokens beyond
  the limit are not trained), empty texts still produce a BOS/EOS example.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from ..data_engine.validation import _as_messages, _msg_content, _msg_role
from ..errors import DatasetEmpty
from ..interfaces import StreamCursor, StreamingDatasetProtocol

log = logging.getLogger(__name__)

SEQ_KEY = "_atk_seq"
PASS_KEY = "pass"
#: Validation rows used for each ``eval_loss`` estimate (a metric sample, not a training cap).
EVAL_MAX_ROWS = 256


@dataclass(frozen=True)
class DataPosition:
    """Position in the training stream: pass (epoch) index + cursor after the last row (None = pass start)."""

    pass_index: int = 0
    cursor: StreamCursor | None = None

    def to_dict(self) -> dict[str, Any] | None:
        """``StreamCursor.to_dict()`` plus a ``"pass"`` key; None at the very beginning."""
        if self.cursor is None and self.pass_index == 0:
            return None
        if self.cursor is None:
            return {"kind": None, "rows_consumed": 0, "state": {}, PASS_KEY: self.pass_index}
        return {**self.cursor.to_dict(), PASS_KEY: self.pass_index}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DataPosition:
        if not d:
            return cls()
        p = int(d.get(PASS_KEY, 0))
        if d.get("kind") is None:
            return cls(p, None)
        return cls(p, StreamCursor.from_dict(d))


class PositionRegistry:
    """Maps example sequence numbers to :class:`DataPosition`; tracks the last *trained* position."""

    def __init__(self, start: DataPosition | None = None) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self._pending: dict[int, DataPosition] = {}
        self.last_consumed: DataPosition = start or DataPosition()
        self.rows_consumed = 0  # training rows consumed in this attempt

    def register(self, pos: DataPosition) -> int:
        with self._lock:
            seq = self._next
            self._next += 1
            self._pending[seq] = pos
            return seq

    def consume(self, seq: int, n_rows: int = 0) -> None:
        """Mark everything up to ``seq`` as trained."""
        with self._lock:
            pos = self._pending.get(seq)
            if pos is None:
                return
            self.last_consumed = pos
            self.rows_consumed += n_rows
            for k in [k for k in self._pending if k <= seq]:
                del self._pending[k]


def chat_messages(row: dict[str, Any], dataset: StreamingDatasetProtocol) -> list[dict[str, str]] | None:
    """Normalised ``[{"role","content"}]`` for chat-format rows, else None."""
    info = dataset.info
    if info.detected_format != "chat" or not info.text_columns:
        return None
    msgs = _as_messages(row.get(info.text_columns[0]))
    if not msgs:
        return None
    out = []
    for m in msgs:
        content = _msg_content(m)
        if content is None:
            continue
        out.append({"role": _msg_role(m) or "user", "content": content})
    return out or None


def make_encoder(dataset: StreamingDatasetProtocol, tokenizer: Any, max_seq_length: int) -> Callable[[dict[str, Any]], list[int]]:
    """Row -> token ids. Chat rows use the tokenizer chat template when it has one; other rows use
    ``dataset.to_text`` + BOS (tokenizer default) + EOS. Truncates to ``max_seq_length``."""
    has_template = bool(getattr(tokenizer, "chat_template", None))
    eos = tokenizer.eos_token_id

    def encode(row: dict[str, Any]) -> list[int]:
        msgs = chat_messages(row, dataset) if has_template else None
        if msgs is not None:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
                ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_seq_length)["input_ids"]
                return list(ids)
            except Exception as e:  # noqa: BLE001 - template incompatible with these roles: fall back to plain text
                log.debug("chat template failed, using plain text: %s", e)
        ids = list(tokenizer(dataset.to_text(row), truncation=True, max_length=max_seq_length)["input_ids"])
        if eos is not None and (not ids or ids[-1] != eos):
            if len(ids) >= max_seq_length:
                ids[-1] = eos
            else:
                ids.append(eos)
        return ids

    return encode


class TrainStream:
    """Endless generator over the train split (re-iterating the source per epoch) for ``from_generator``.

    Stops (yields nothing more) when ``stop_event`` is set or the dataset was cancelled.
    """

    def __init__(self, dataset: StreamingDatasetProtocol, encode: Callable[[dict[str, Any]], list[int]],
                 registry: PositionRegistry, start: DataPosition, val_fraction: float, seed: int,
                 stop_event: threading.Event | None = None) -> None:
        self.dataset = dataset
        self.encode = encode
        self.registry = registry
        self.start = start
        self.val_fraction = val_fraction
        self.seed = seed
        self.stop_event = stop_event

    def _stopped(self) -> bool:
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def __call__(self) -> Iterator[dict[str, Any]]:
        pass_index, cursor = self.start.pass_index, self.start.cursor
        while not self._stopped():
            n = 0
            for row, cur in self.dataset.iter_rows(cursor, split="train", val_fraction=self.val_fraction,
                                                   seed=self.seed):
                n += 1
                seq = self.registry.register(DataPosition(pass_index, cur))
                yield {"input_ids": self.encode(row), SEQ_KEY: seq}
                if self._stopped():
                    return
            if getattr(self.dataset.stats, "state", None) == "cancelled":
                return
            if n == 0 and cursor is None:
                raise DatasetEmpty(
                    "the training split produced no rows (dataset empty, or val_fraction leaves no training rows)"
                )
            pass_index, cursor = pass_index + 1, None


def make_collator(pad_token_id: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Right-pad to the longest example; labels = input_ids with padding masked to -100.

    Forwards ``_atk_seq`` (max of the batch) and ``_atk_rows`` (batch size) for cursor tracking.
    """
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        longest = max(1, max(len(f["input_ids"]) for f in features))
        ids = torch.full((len(features), longest), int(pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(features), longest), dtype=torch.long)
        for i, f in enumerate(features):
            row = list(f["input_ids"])
            if row:
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, : len(row)] = 1
        batch: dict[str, Any] = {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}
        if SEQ_KEY in features[0]:
            batch[SEQ_KEY] = max(int(f[SEQ_KEY]) for f in features)
            batch["_atk_rows"] = len(features)
        return batch

    return collate


def iter_eval_batches(dataset: StreamingDatasetProtocol, encode: Callable[[dict[str, Any]], list[int]],
                      collate: Callable[[list[dict[str, Any]]], dict[str, Any]], val_fraction: float, seed: int,
                      batch_size: int, max_rows: int = EVAL_MAX_ROWS) -> Iterator[dict[str, Any]]:
    """Batches from the first ``max_rows`` rows of the validation split (deterministic sample)."""
    buf: list[dict[str, Any]] = []
    seen = 0
    gen = dataset.iter_rows(None, split="validation", val_fraction=val_fraction, seed=seed)
    try:
        for row, _cur in gen:
            buf.append({"input_ids": encode(row)})
            seen += 1
            if len(buf) >= batch_size:
                yield collate(buf)
                buf = []
            if seen >= max_rows:
                break
        if buf:
            yield collate(buf)
    finally:
        close = getattr(gen, "close", None)
        if close is not None:
            close()
