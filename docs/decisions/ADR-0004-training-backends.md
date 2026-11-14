# ADR-0004: Training backends

Status: accepted (2026-11-14)

## Decision
- `unsloth` backend when CUDA is available and Unsloth imports: `FastLanguageModel.from_pretrained(load_in_4bit=...)` + `get_peft_model`, trained with TRL `SFTTrainer`. Unsloth kernels are used as-is.
- `hf` backend otherwise: Transformers `AutoModelForCausalLM` + PEFT `LoraConfig`, trained with the HF `Trainer`/TRL on an `IterableDataset` built from the streaming dataset.
- QLoRA (4-bit) requires CUDA + bitsandbytes; requesting it without CUDA is a validation error with guidance (no silent downgrade).
- Checkpoints store adapter weights + `autotune_state.json` (step, data cursor, configs).

## Verification
The `hf` backend is verified on CPU with a tiny Llama model. The `unsloth` path can only be verified on a CUDA runtime (Colab/Kaggle notebooks) and is reported UNVERIFIED until run there.
