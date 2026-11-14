"""Row validation, integrity checksum and training-text formatting."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..contracts import DatasetInfo

MAX_MALFORMED_EXAMPLES = 5

_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}


def canonical_json(row: Any) -> str:
    """Canonical JSON used for checksums and the hash split (sorted keys, ``default=str``)."""
    return json.dumps(row, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))


def _as_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):  # column may have been coerced to JSON strings
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    if not all(isinstance(m, dict) for m in value):
        return None
    return value


def _msg_role(m: dict[str, Any]) -> str | None:
    role = m.get("role") if m.get("role") is not None else m.get("from")
    return None if role is None else _ROLE_ALIASES.get(str(role).lower(), str(role))


def _msg_content(m: dict[str, Any]) -> str | None:
    c = m.get("content") if m.get("content") is not None else m.get("value")
    if c is None:
        return None
    if isinstance(c, list):  # multimodal-style content parts: keep text parts
        c = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def to_text(row: dict[str, Any], info: DatasetInfo) -> str:
    """Format ``row`` into training text according to ``info.detected_format``.

    * chat: role-tagged transcript ``"<|role|>\\ncontent"`` per message (training
      applies the tokenizer chat template instead when one is available);
    * instruction: Alpaca-style prompt (``### Instruction / ### Input / ### Response``);
    * text: the text column;
    * unknown: canonical JSON of the row (a warning is recorded at discovery).
    """
    fmt, cols = info.detected_format, info.text_columns
    if fmt == "chat" and cols:
        msgs = _as_messages(row.get(cols[0])) or []
        parts = []
        for m in msgs:
            content = _msg_content(m)
            if content is None:
                continue
            parts.append(f"<|{_msg_role(m) or 'user'}|>\n{content}")
        return "\n".join(parts)
    if fmt == "instruction" and len(cols) >= 2:
        instr, out = _s(row.get(cols[0])), _s(row.get(cols[-1]))
        inp = _s(row.get(cols[1])) if len(cols) == 3 else ""
        text = f"### Instruction:\n{instr}\n\n"
        if inp.strip():
            text += f"### Input:\n{inp}\n\n"
        return text + f"### Response:\n{out}"
    if fmt == "text" and cols:
        return _s(row.get(cols[0]))
    return canonical_json(row)


def malformed_reason(row: dict[str, Any], info: DatasetInfo) -> str | None:
    """Return why ``row`` cannot be used for training, or None if it is fine."""
    fmt, cols = info.detected_format, info.text_columns
    if fmt == "chat" and cols:
        msgs = _as_messages(row.get(cols[0]))
        if msgs is None:
            return f"column {cols[0]!r} is not a list of messages"
        if not any((_msg_content(m) or "").strip() for m in msgs):
            return "conversation has no message content"
        return None
    if fmt == "instruction" and len(cols) >= 2:
        if not _s(row.get(cols[0])).strip() and not _s(row.get(cols[-1])).strip():
            return f"empty {cols[0]!r} and {cols[-1]!r}"
        if not _s(row.get(cols[-1])).strip():
            return f"empty {cols[-1]!r}"
        return None
    if fmt == "text" and cols:
        v = row.get(cols[0])
        if v is None or not str(v).strip():
            return f"empty text in column {cols[0]!r}"
        return None
    if not row or all(v is None for v in row.values()):
        return "row has no values"
    return None


class RowValidator:
    """Validates rows of one streaming pass, counting (never hiding) malformed ones.

    ``checksum`` is a rolling blake2b over the canonical JSON of every accepted
    row, so two passes over the same rows produce the same digest.
    """

    def __init__(self, info: DatasetInfo, max_examples: int = MAX_MALFORMED_EXAMPLES) -> None:
        self.info = info
        self.max_examples = max_examples
        self.accepted_rows = 0
        self.malformed_rows = 0
        self.malformed_examples: list[str] = []
        self._hash = hashlib.blake2b(digest_size=16)

    def record_malformed(self, description: str) -> None:
        """Count one malformed record and keep up to ``max_examples`` descriptions."""
        self.malformed_rows += 1
        if len(self.malformed_examples) < self.max_examples:
            self.malformed_examples.append(description[:300])

    def check(self, row: dict[str, Any], location: str = "") -> bool:
        """Validate ``row``; on success update the checksum and return True."""
        reason = malformed_reason(row, self.info)
        if reason is not None:
            snippet = canonical_json(row)
            snippet = snippet if len(snippet) <= 200 else snippet[:200] + "..."
            self.record_malformed(f"{location + ': ' if location else ''}{reason}: {snippet}")
            return False
        self.accept(row)
        return True

    def accept(self, row: dict[str, Any]) -> None:
        """Fold an accepted row into the rolling checksum."""
        self.accepted_rows += 1
        self._hash.update(canonical_json(row).encode("utf-8"))
        self._hash.update(b"\n")

    @property
    def checksum(self) -> str:
        return self._hash.hexdigest()


def ensure_usable(accepted_rows: int, malformed_rows: int, malformed_examples: list[str], source: str) -> None:
    """Raise :class:`DatasetEmpty` (no records at all) or :class:`DatasetMalformed` (only bad records)."""
    from ..errors import DatasetEmpty, DatasetMalformed

    if accepted_rows > 0:
        return
    if malformed_rows == 0:
        raise DatasetEmpty(f"dataset {source} contains no rows", {"source": source})
    raise DatasetMalformed(
        f"all {malformed_rows} examined rows of {source} are malformed",
        {"source": source, "malformed_rows": malformed_rows, "examples": malformed_examples},
    )


def rows_checksum(rows: Any) -> str:
    """Checksum of an iterable of rows, identical to :attr:`RowValidator.checksum` for the same rows."""
    h = hashlib.blake2b(digest_size=16)
    for row in rows:
        h.update(canonical_json(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
