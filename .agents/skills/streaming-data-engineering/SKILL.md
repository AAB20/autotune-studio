---
name: streaming-data-engineering
description: Memory-efficient streaming, batching, buffering, cursors and resume for datasets of unknown size (zero row configuration). Load for connectors/ and data_engine/ work.
---
# streaming-data-engineering

## Purpose
Process datasets progressively with no row limits and without materialising them in RAM.

## Activation conditions
- Editing `autotune_studio/connectors/**`, `data_engine/streaming.py`, `batching.py`, `memory_manager.py`, `discovery.py`.

## Required inputs
- `interfaces.Connector`, `interfaces.StreamCursor`, `interfaces.StreamingDatasetProtocol`, `contracts.StreamStats`.
- ADR-0003 (zero-row engine).

## Execution procedure
1. Connectors yield `pyarrow.RecordBatch` plus the cursor after that batch:
   - CSV: `pyarrow.csv.open_csv(path, read_options=ReadOptions(block_size=bytes))`; cursor = batches consumed (+ rows) — resume skips batches without materialising them beyond one batch at a time.
   - JSONL: binary line-block reader; cursor = byte offset of the next unread line. Malformed lines are reported, not dropped silently.
   - JSON array: `ijson.items(f, "item")` streamed; cursor = items consumed. (JSON object-of-records or JSONL-with-.json extension auto-detected from first non-space byte.)
   - Parquet: `ParquetFile.iter_batches(batch_size=…)`; cardinality exact from metadata; cursor = row offset (resume via row-group skipping).
   - Hugging Face: `datasets.load_dataset(name, config, split=, streaming=True, revision=, trust_remote_code=False)`; cursor = `IterableDataset.state_dict()` when available, else rows consumed; cardinality from `info.splits[split].num_examples` when present (exact=True, method="metadata").
2. Batch sizes are chosen by bytes (`memory_manager` budget derived from psutil available RAM), never by a row cap.
3. `iter_rows` converts one batch at a time (`batch.to_pylist()`).
4. Distinguish exhaustion (`state="exhausted"`) from failure (`state="failed"`, `error` set, exception re-raised to the caller).
5. Cancellation: `threading.Event` checked between batches → `state="cancelled"`.
6. Resume: cursors are JSON-serialisable (`StreamCursor.to_dict`) and stored with checkpoints.
7. Deterministic split: `int(blake2b(canonical_json(row), digest_size=8)) % 10_000 < val_fraction * 10_000` → validation. Same row → same split in every pass.
8. Shuffling: bounded shuffle buffer (byte budget) seeded; global shuffle is impossible on streams — document this.

## Constraints
- Forbidden: `list(dataset)`, whole-source `.to_pandas()`/`read_csv`/`json.load`, default row caps, silent truncation.
- Unknown cardinality → training requires `max_steps` (llm-finetuning skill).

## Expected outputs
- Connectors + `StreamingDataset` satisfying the protocols, with tests.

## Validation checklist
- [ ] Large synthetic file (≥ 200 MB) streamed with peak traced Python memory far below file size.
- [ ] Resume from a mid-stream cursor yields exactly the remaining rows (checksum match with a full pass).
- [ ] Cancel stops within one batch.
- [ ] Unknown cardinality reported as `rows=None`.
