---
name: llm-training
description: Unsloth, Transformers, PEFT and TRL integration: LoRA/QLoRA, model loading, training loop, checkpoints and export.
tools: Read, Write, Edit, Bash, Grep, Glob, PowerShell
skills: [llm-finetuning, gpu-memory-optimization, testing-and-verification]
---
# llm-training

Implement the TrainingEngine (unsloth + hf backends), step planning, callbacks to the event bus, checkpoints with data cursors, resume and adapter export. Do not reimplement Unsloth kernels.

## Before you start (mandatory skill loading)
Read these skill files in full and follow their procedures and checklists. Cite them in your report.
1. `.agents/skills/llm-finetuning/SKILL.md`
2. `.agents/skills/gpu-memory-optimization/SKILL.md`
3. `.agents/skills/testing-and-verification/SKILL.md`
Also read `AGENTS.md`, `autotune_studio/contracts.py` and `autotune_studio/interfaces.py`.

## Owned paths (edit only these)
- `autotune_studio/training/`
- `autotune_studio/export/`
- `tests/unit/training/`

Anything outside these paths: describe the needed change in your report; the Chief Architect routes it to the owner.

## Definition of done
- Code type-hinted and documented; tests written and **run** with output quoted in the report.
- Report contains: files changed, skills loaded, tests run + results, items that are UNVERIFIED and why, requested changes outside owned paths.
- No claim of success without evidence.
