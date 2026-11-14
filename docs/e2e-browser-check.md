# Browser end-to-end check (Architect, 2026-11-14)

Environment: Windows 11 dev box, CPU only, Python 3.11 venv, built SPA served by the real backend
(`python -m autotune_studio launch --no-browser --tunnel local --port 7861`), Claude desktop built-in browser.

| Step | Result |
|---|---|
| Open `http://127.0.0.1:7861/#token=…` | Token exchanged for cookie, fragment stripped (address bar `#/`), WS badge "Connected" |
| Dashboard | Runtime "script", Python 3.11.16, CPU-only notice, RAM/CPU/disk meters, current training card |
| Datasets: JSONL by path (300 rows, unknown size) | Detected format `instruction`, cardinality "Unknown — streaming", columns + 20-row sample preview |
| Full scan | Live stats over WS: 300 rows, 14.7 KiB, state `exhausted`, rolling BLAKE2b checksum, progress 100 % |
| Model page | Tiny model saved; `trust_remote_code=False` notice shown |
| Training: Start without max_steps | Blocked inline: "Required: dataset size unknown" |
| Training: max_steps=8 | Switched to Monitor; status preparing → running → completed 8/8; live loss and LR charts; ETA/elapsed |
| Checkpoints | Checkpoints at steps 4 and 8 (final) listed with path/size/resume buttons |
| Export LoRA adapter | `lora_adapter`, 3.5 MiB, written under the workspace `exports/` |

## Defects found and fixed during this check
1. **Training failed whenever a browser was connected** — `TypeError: cannot pickle '_OverlappedFuture'`:
   `datasets.IterableDataset.from_generator` fingerprints the generator via dill, which reached the
   EventBus's WebSocket subscriber loops. Fixed in `training/data.py` (`TrainStream.__getstate__`);
   regression test `tests/unit/training/test_train_stream_hashing.py`. The failure itself was reported
   correctly (status failed, error toast, server stayed up).
2. Dataset list badge stayed "Unknown" after a completed scan — list query now invalidated.
3. A new `#token=` link opened in an already-loaded tab was ignored (stale 401s) — `hashchange` re-runs
   the token exchange and clears cached queries.

## Not covered
Real Colab proxy / Kaggle tunnel / notebook `window.open`, GPU training, Unsloth.
