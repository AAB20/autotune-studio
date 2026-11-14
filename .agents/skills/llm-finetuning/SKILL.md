---
name: llm-finetuning
description: Unsloth, Transformers, PEFT and TRL integration for LoRA/QLoRA, model loading, step/epoch planning, checkpoints and adapter export. Load for training/ and export/.
---
# llm-finetuning

## Purpose
Run a real, verifiable LoRA/QLoRA fine-tune on streamed data: Unsloth on CUDA, a PEFT + Transformers/TRL fallback elsewhere.

## Activation conditions
- Editing `autotune_studio/training/**`, `autotune_studio/export/**`.

## Required inputs
- `ModelConfig`, `TrainingConfig`, `interfaces.TrainingEngineProtocol`, `StreamingDatasetProtocol`, `TrainingControl`, `EventSink`.
- ADR-0004 (training backends).

## Execution procedure
1. Backend selection (`training/trainer.select_backend`): `unsloth` if CUDA is available and `import unsloth` succeeds; else `hf`.
2. QLoRA needs CUDA + bitsandbytes. On CPU, `method="qlora"` raises `errors.ValidationFailed` explaining that QLoRA requires a CUDA GPU and suggesting `method="lora"` — never silently downgrade.
3. Import `unsloth` before `transformers` when used. `trust_remote_code=False` always.
4. Dataset → `datasets.IterableDataset.from_generator` over `dataset.iter_rows(split="train", val_fraction, seed)` mapped through `dataset.to_text()`; tokenisation streams (truncate to `max_seq_length`). Apply the tokenizer chat template for chat data when the tokenizer has one.
5. Step planning (`training/trainer.plan_steps`): if `cardinality.rows` known → `ceil(rows*(1-val_fraction) / (micro*accum*world)) * num_epochs`; unknown → `max_steps` required, else `errors.StepsRequired`. Explicit `max_steps` always wins. Streaming datasets loop for multiple epochs by re-iterating the source.
6. Callbacks publish `metric` and `status` events and honour `control.stop`, `control.pause` (blocks between steps), `control.save_now`.
7. Checkpoints (`training/checkpoints.py`): adapter + tokenizer via `save_pretrained` into `output_dir/checkpoint-<step>/` plus `autotune_state.json` (step, data cursor, configs, job id). Final checkpoint flagged `is_final`.
8. Resume: load adapter weights from checkpoint and continue from the saved data cursor/step.
9. Export (`export/model_export.py`): `lora_adapter` copies/saves adapter files; `merged_16bit` via Unsloth `save_pretrained_merged` or PEFT `merge_and_unload().save_pretrained`.
10. Verify the APIs of the installed versions (transformers 5.x, trl 1.x, peft 0.21) — signatures changed vs. older tutorials (e.g. `processing_class` instead of `tokenizer`, `SFTConfig` fields).

## Constraints
- Routine tests use tiny models (e.g. `hf-internal-testing/tiny-random-LlamaForCausalLM`, marked `network` on first download) or locally constructed tiny configs (`LlamaConfig(hidden_size=32, …)` + a small tokenizer) so CI needs no Hub access.
- Do not reimplement optimisation kernels.
- Never report success before the trainer returned and checkpoint files exist on disk.

## Expected outputs
- `TrainingResult`, checkpoints on disk, `ExportResult`.

## Validation checklist
- [ ] CPU LoRA run on a tiny model: loss events, checkpoint saved, adapter exported and reloadable with PEFT.
- [ ] Unknown length without `max_steps` → `StepsRequired`.
- [ ] Stop mid-run → state `stopped`, checkpoint saved.
- [ ] Unsloth path marked UNVERIFIED unless run on a GPU.
