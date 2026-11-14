# ADR-0003: Zero row configuration engine

Status: accepted (2026-11-14)

## Decision
- Users never supply row counts. All sources are streamed as Arrow record batches sized by a **byte budget** derived from available RAM (`data_engine/memory_manager.py`).
- Schema/format discovery samples a bounded number of **bytes** (default 4 MiB), not rows.
- Cardinality: exact from metadata (Parquet footer, HF split info) when available; otherwise `rows=None` (unknown). An optional explicit full streaming scan (`POST /api/datasets/{id}/scan`) counts rows and validates every row without keeping them.
- Cursors (`StreamCursor`) make streams resumable and are stored with checkpoints.
- Malformed rows are counted and reported, never silently dropped; empty datasets raise `DatasetEmpty`; failures are distinct from exhaustion.
- Train/validation split is a deterministic hash of the canonical row (no global shuffle needed). Shuffling uses a bounded, seeded buffer.
- Training: known length → steps from epochs; unknown length → `max_steps` required (`StepsRequired` error otherwise). Multiple epochs over streams re-iterate the source.

## Rejected
Row caps / `take(n)` defaults; loading into pandas; global shuffle of streams (impossible without materialisation).
