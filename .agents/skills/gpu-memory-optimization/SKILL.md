---
name: gpu-memory-optimization
description: VRAM/RAM monitoring, adaptive batch sizing, OOM detection and safe recovery without dropping data. Load for monitoring/ and OOM handling in training/.
---
# gpu-memory-optimization

## Purpose
Keep training inside memory limits and recover from OOM without losing or skipping data.

## Activation conditions
- Editing `autotune_studio/monitoring/**`, OOM handling in `training/`, memory budgets in `data_engine/memory_manager.py`.

## Required inputs
- `contracts.ResourceSnapshot`, `HardwareInfo`, `TrainingConfig.adaptive_batch_size`, `errors.OutOfMemory`.

## Execution procedure
1. `monitoring/memory.py`: psutil RAM + CPU%. `monitoring/gpu.py`: torch.cuda `memory_allocated`/`memory_reserved`/`get_device_properties().total_memory` (lazy torch import); NVML utilisation when `pynvml` imports. `ResourceMonitor.snapshot()` merges both; all GPU fields None without CUDA.
2. `monitoring/metrics.py`: `ResourceSampler` background thread publishing `resource` events every `interval` seconds (default 2) to an `EventSink`; `start()`/`stop()` idempotent; `MetricsStore` keeps the training metric history (bounded deque, downsampling older points) for `GET /api/training/metrics`.
3. `suggest_micro_batch(free_vram, seq_len, params)`: documented heuristic estimate only.
4. OOM policy (`monitoring/oom.py`, used by the trainer): detect `torch.cuda.OutOfMemoryError` or RuntimeError containing "out of memory"; `gc.collect(); torch.cuda.empty_cache()`; new micro = micro // 2, accum = accum * 2 (effective batch preserved); retry from the last checkpoint + data cursor; at micro == 1 raise `errors.OutOfMemory`.
5. Every adaptation publishes a `log` event (level warning) with old/new values.

## Constraints
- Never discard samples to relieve memory pressure.
- Monitoring must work without CUDA/NVML and must never crash training.

## Expected outputs
- `monitoring/gpu.py`, `memory.py`, `metrics.py`, `oom.py` + tests.

## Validation checklist
- [ ] Snapshot works on a CPU-only machine.
- [ ] OOM policy unit test: halving/doubling sequence, terminal error at 1.
- [ ] Sampler thread stops within one interval.
