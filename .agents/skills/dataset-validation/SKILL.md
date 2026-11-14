---
name: dataset-validation
description: Schema discovery, format detection (chat/instruction/text), malformed-row handling, integrity checksums and empty-dataset errors. Load for data_engine/discovery.py and validation.py.
---
# dataset-validation

## Purpose
Understand a dataset from a bounded sample and validate every streamed row without losing or hiding data.

## Activation conditions
- Editing `data_engine/discovery.py`, `data_engine/validation.py`; adding a format.

## Required inputs
- `contracts.ColumnInfo`, `DatasetInfo`, `StreamStats`; `errors.DatasetEmpty`, `DatasetMalformed`.

## Execution procedure
1. Sample by byte budget (default 4 MiB), not by row count; record `sampled_rows`, `sampled_bytes`.
2. Columns: arrow dtype string, sampled null count, up to 3 examples truncated to 120 chars.
3. Format detection:
   - chat: column `messages` / `conversations` holding a list of {role|from, content|value}.
   - instruction: `instruction` + `output` (+ optional `input`), or `prompt` + `response`/`completion`/`answer`, or `question` + `answer`.
   - text: a dominant string column (`text`, `content`, `document`, or the only/longest string column).
   - else unknown + warning; user may set `text_column`.
4. `to_text(row)`: chat → simple role-tagged transcript (tokenizer chat templates applied later by training when available); instruction → Alpaca-style prompt; text → the column.
5. Malformed rows (parse errors, wrong types, empty text): increment `malformed_rows`, keep ≤5 examples, continue streaming. If every sampled row is malformed → `DatasetMalformed`.
6. Empty source (zero rows) → `DatasetEmpty` — distinct from exhaustion after ≥1 row.
7. Integrity: rolling blake2b over canonical JSON (sorted keys, `default=str`) of each accepted row → `checksum`.

## Constraints
- Never drop rows silently: every skipped row is counted and visible in stats.
- Never execute dataset content (no `eval`, `pickle`, dataset loading scripts).

## Expected outputs
- `DatasetInfo` + live `StreamStats`.

## Validation checklist
- [ ] Tests: empty file, header-only CSV, malformed JSONL lines, mixed types, all-malformed sample.
- [ ] Format detection tests for chat / instruction / text / unknown.
- [ ] Checksum identical across two full passes and full-vs-resumed passes.
