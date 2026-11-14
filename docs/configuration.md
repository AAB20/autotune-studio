# Configuration

## Environment variables

Every `AUTOTUNE_*` variable that the package reads, found by grepping `autotune_studio/` and
`tests/` on 2026-11-14:

### Server (`autotune_studio/core/config.py`, `load_settings()`)

Read when the FastAPI app is created (`Studio.launch()` / `create_app()`). Invalid values raise
`ValueError` naming the variable. Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`.

| Variable | Default | Effect |
|---|---|---|
| `AUTOTUNE_UPLOAD_MAX_BYTES` | `2147483648` (2 GiB) | Maximum dataset upload size (`POST /api/datasets/upload`). |
| `AUTOTUNE_CORS_ENABLED` | `false` | Turns on CORS. It is off by default because the UI is served from the same origin. |
| `AUTOTUNE_CORS_ORIGINS` | empty | Comma-separated allowed origins, used only when CORS is enabled. Credentials are not allowed cross-origin, so use Bearer. |
| `AUTOTUNE_WS_PING_INTERVAL` | `20` (seconds) | WebSocket keepalive interval. |
| `AUTOTUNE_REPLAY_SIZE` | `2000` | Size of the event replay buffer for WebSocket reconnects. Read in `Studio()` when `replay_size` is not passed. |
| `AUTOTUNE_LOG_LEVEL` | `info` | Parsed and validated (`debug`/`info`/`warning`/`error`/`critical`), but **not applied** anywhere in 0.1.0. The server thread logs at `warning`. |
| `AUTOTUNE_HOST` | `127.0.0.1` | Parsed and validated, but **not used** by `Studio.launch()` in 0.1.0. Use `launch(host=...)` / `--host`. |
| `AUTOTUNE_PORT` | `7860` | Parsed and validated, but **not used** by `Studio.launch()` in 0.1.0. Use `launch(port=...)` / `--port`. Without either, the server tries 7860..7879 and then an OS-assigned port. |

### Runtime (`autotune_studio/core/environment.py`)

| Variable | Effect |
|---|---|
| `AUTOTUNE_WORKSPACE` | Overrides the default workspace (Colab `/content/autotune`, Kaggle `/kaggle/working/autotune`, local `<cwd>/.autotune`). `Studio(workspace_dir=...)` and `--workspace` take precedence. |
| `AUTOTUNE_ALLOWED_ROOTS` | Extra directories datasets may be read from, separated by `os.pathsep` (`;` on Windows, `:` elsewhere). Non-existent entries are kept and a note is added. |

### Tunnel (`autotune_studio/launch/cloudflared.py`)

| Variable | Effect |
|---|---|
| `AUTOTUNE_CLOUDFLARED` | Explicit path to a `cloudflared` binary. It is trusted and not hashed. Otherwise `PATH` is used, then a SHA256-verified download of the pinned release `2026.11.0`. |
| `AUTOTUNE_CACHE_DIR` | Base directory for the downloaded binary (`<dir>/cloudflared`). Default: `%LOCALAPPDATA%\autotune_studio\cloudflared` on Windows, `${XDG_CACHE_HOME:-~/.cache}/autotune_studio/cloudflared` elsewhere. |

### Training (`autotune_studio/training/planning.py`)

| Variable | Effect |
|---|---|
| `AUTOTUNE_TRAINING_BACKEND` | `hf` forces the Transformers+PEFT backend. `unsloth` forces Unsloth, which fails with `dependency_missing` if Unsloth is not installed. Unset: Unsloth when CUDA is available and `unsloth` is importable, otherwise `hf`. |

### Tests only (`tests/`)

| Variable | Effect |
|---|---|
| `AUTOTUNE_RUN_NETWORK=1` | Runs tests marked `network` (Hub access). Same as `pytest --run-network`. |
| `AUTOTUNE_TUNNEL_SMOKE=1` | Enables `tests/unit/runtime/test_tunnel_smoke.py`, which opens a real public quick tunnel. |
| `AUTOTUNE_TUNNEL_DOWNLOAD=1` | Lets that smoke test download cloudflared. |

### Other variables the package reads

| Variable | Effect |
|---|---|
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hugging Face token, checked before the Colab `userdata` / Kaggle secret named `HF_TOKEN`. Never logged. |
| `COLAB_RELEASE_TAG` | Its presence means Colab. |
| `KAGGLE_KERNEL_RUN_TYPE` | Its presence means Kaggle. `batch` adds a "no interactive browser" note. |
| `CUDA_VISIBLE_DEVICES` | Used to map torch device indices to NVML GPUs. |

## `ModelConfig`

Set with `studio.configure_model(name, **fields)` or `PUT /api/model`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_name` | str | required | Hugging Face repo id (`owner/name` or a legacy bare id like `gpt2`), or a local directory inside an allowed root. No URLs, no `..`. |
| `revision` | str \| None | `None` | Hub revision |
| `max_seq_length` | int | `2048` | 16..131072 |
| `load_in_4bit` | bool | `True` | Used by QLoRA. `method="lora"` ignores it (with a warning). |
| `dtype` | `auto` \| `float16` \| `bfloat16` \| `float32` | `auto` | `bfloat16` on a GPU without bf16 is rejected. On CPU, training runs in float32. |
| `trust_remote_code` | `False` | `False` | Cannot be enabled. |

Suggested models (`studio.suggested_models()`): `unsloth/Llama-3.2-1B-Instruct`,
`unsloth/Llama-3.2-3B-Instruct`, `unsloth/Qwen2.5-0.5B-Instruct`, `unsloth/Qwen2.5-1.5B-Instruct`,
`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, `unsloth/gemma-2-2b-it`,
`unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, `unsloth/Phi-3.5-mini-instruct`.

## `TrainingConfig`

Set with `studio.configure_training(method, **fields)` (this merges into the current config) or
`PUT /api/training/config` (this replaces it). The facade's initial `method` is `qlora` when CUDA
is available and `lora` otherwise. The model default is `qlora`.

| Field | Type | Default | Constraint / notes |
|---|---|---|---|
| `method` | `lora` \| `qlora` | `qlora` | QLoRA needs CUDA and bitsandbytes or unsloth. |
| `learning_rate` | float | `2e-4` | 0 < x <= 1 |
| `num_epochs` | float \| None | `1.0` | > 0. Used only when the dataset length is known. |
| `max_steps` | int \| None | `None` | > 0. **Required when the length is unknown.** Overrides epochs when set. |
| `per_device_batch_size` | int | `2` | 1..1024 (the micro-batch) |
| `gradient_accumulation_steps` | int | `4` | 1..4096 |
| `lora_r` | int | `16` | 1..1024 |
| `lora_alpha` | int | `16` | 1..4096 |
| `lora_dropout` | float | `0.0` | 0 <= x < 1 |
| `target_modules` | list[str] \| None | `None` | `None` = backend default |
| `optimizer` | `adamw_torch` \| `adamw_8bit` \| `paged_adamw_8bit` \| `sgd` \| `adafactor` | `adamw_torch` | The 8-bit optimizers need CUDA and bitsandbytes. |
| `lr_scheduler` | `linear` \| `cosine` \| `constant` \| `constant_with_warmup` | `linear` | |
| `warmup_steps` | int | `5` | >= 0. A warning is given if it is >= the total steps. |
| `weight_decay` | float | `0.0` | >= 0 |
| `logging_steps` | int | `1` | >= 1 |
| `save_steps` | int | `50` | >= 1. A final checkpoint is always written. |
| `seed` | int | `3407` | |
| `val_fraction` | float | `0.0` | 0..0.5. Deterministic hash-based split. `eval_loss` is estimated on up to 256 validation rows per checkpoint. |
| `output_dir` | str \| None | `None` | Default `<workspace>/runs/<job_id>`. Must be inside an allowed root. |
| `adaptive_batch_size` | bool | `True` | On CUDA OOM, halve the micro-batch, raise accumulation to keep the effective batch, and resume from the last checkpoint. |

Step planning (`training/planning.py`):

- With `max_steps` set, that value is used.
- With a known row count, the step count is
  `ceil(ceil(rows * (1 - val_fraction) / (batch * accum)) * num_epochs)`. Example (verified
  locally): 40 rows, batch 2, accumulation 4, 1 epoch → 5 steps.
- With an unknown row count and no `max_steps`, you get `422 max_steps_required`.
