# AutoTune Studio

AutoTune Studio fine-tunes large language models with LoRA or QLoRA. It runs inside Google Colab,
Kaggle, local Jupyter or a plain Python script. One call, `Studio().launch()`, starts a FastAPI
backend in a background thread of your kernel and opens a React dashboard in a **separate browser
tab**. From there you pick a dataset and a model, then train, watch live metrics, checkpoint,
resume and export. Everything in the UI is also available from Python.

> Status: alpha (0.1.0). Verified on a Windows CPU dev box only. Colab, Kaggle and GPU training
> (Unsloth / QLoRA) are **UNVERIFIED**; see [Status](#status).

## Zero row configuration

You never tell AutoTune Studio how many rows a dataset has, and it never caps or samples rows for
training:

- Every source (Hugging Face Hub, CSV, JSON, JSONL, Parquet) is **streamed** as Arrow record batches
  sized by a memory budget. Nothing is loaded into memory as a whole.
- Schema and format (chat / instruction / text) are detected from a byte-bounded sample.
- The row count is taken from metadata when the source has it (Parquet footer, Hugging Face split
  info). Otherwise it is reported as **unknown** (`cardinality.rows is None`). You can resolve it
  with an optional full streaming scan, which counts and validates every row without keeping them.
- Training converts epochs to steps when the length is known. When it is unknown you set
  `max_steps`; the API answers `422 max_steps_required` until you do.
- Malformed rows are counted and reported, never dropped silently.

Background: [ADR-0003](docs/decisions/ADR-0003-zero-row-engine.md).

## Features

- Runtime detection (Colab / Kaggle / Jupyter / script) with a matching workspace and dataset
  allowlist, plus hardware detection (CPU, RAM, disk, GPUs, CUDA, bf16).
- UI exposure that fits the runtime: the Colab kernel proxy, a cloudflared quick tunnel (pinned
  version, SHA256-verified) or local only. A public URL counts as verified only after a health
  probe through it succeeds.
- Datasets from the Hugging Face Hub (streaming) or local CSV/TSV, JSON, JSONL/NDJSON and Parquet
  files, by path or browser upload. You get a preview, live streaming statistics and a rolling
  checksum.
- Training backends: Unsloth (CUDA, used automatically when installed) or Hugging Face
  Transformers + PEFT (CPU or GPU). Methods: LoRA and QLoRA (4-bit, CUDA only).
- Live loss, learning-rate and resource charts over a WebSocket that resumes from the last event
  after a reconnect.
- Stop (a checkpoint is saved first), pause/resume, save-now and resume from any checkpoint. The
  data cursor is stored too, so a resumed run continues at the exact row.
- On CUDA out-of-memory the micro-batch is halved and gradient accumulation raised, so the
  effective batch is kept (rounded up). Training then resumes from the last checkpoint.
- Export a LoRA adapter, or a merged 16-bit model.
- Security: a per-launch session token in the URL fragment, an HttpOnly cookie, every API/WS route
  authenticated, path allowlists, `trust_remote_code` always `False` and secrets redacted from logs.

## Quickstart

There is no PyPI release yet. Build a wheel (see [installation](docs/installation.md#build-a-wheel))
or install from this repository (`https://github.com/AAB20/autotune-studio.git`). The built browser
UI is committed under `autotune_studio/web/`, so both a wheel and a `git+` install include it.

The repository is **private**: a notebook needs a read-only GitHub token to install it. Create a
fine-grained token with *Contents: Read* on this repository and store it as the secret `GITHUB_TOKEN`
(Colab: *Secrets* panel; Kaggle: *Add-ons > Secrets*). The notebooks read it without printing it.

### Google Colab (UNVERIFIED)
Open [`notebooks/colab.ipynb`](notebooks/colab.ipynb), select a T4 GPU runtime and run the cells in
order. The install cell installs from the repository (needs `GITHUB_TOKEN`) or an uploaded wheel. Details: [docs/colab.md](docs/colab.md).

### Kaggle (UNVERIFIED)
Open [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb). In *Settings*, choose a GPU accelerator and
switch **Internet ON** (this needs a phone-verified account). Install from the repository (needs the
`GITHUB_TOKEN` secret) or add the wheel as a Kaggle dataset. Details: [docs/kaggle.md](docs/kaggle.md).

### Local (verified on Windows, CPU)
```bash
uv venv .venv -p 3.11
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -e ".[hf,train,dev]"
cd frontend && npm ci && npm run build && cd ..          # builds the UI into autotune_studio/web/
.venv/Scripts/python.exe -m autotune_studio env           # runtime + hardware as JSON
.venv/Scripts/python.exe -m autotune_studio launch        # opens the UI in a new tab; Ctrl+C stops
```
(On Linux/macOS use `.venv/bin/python`.) Full guide: [docs/installation.md](docs/installation.md).

## Python API

```python
from autotune_studio import Studio

studio = Studio()                       # detects runtime, workspace and allowed roots
studio.launch()                         # optional: the UI in a separate tab

info = studio.load_dataset("jsonl", path="data/train.jsonl")   # or ("huggingface", name="yahma/alpaca-cleaned")
print(info.detected_format, info.cardinality.rows)             # rows=None -> unknown, set max_steps

studio.configure_model("unsloth/Llama-3.2-1B-Instruct")
studio.configure_training(method="qlora", max_steps=60)       # method="lora" on CPU
studio.start_training()
status = studio.wait()
result = studio.export_model()          # LoRA adapter from the latest checkpoint
studio.shutdown()
```

This sequence was run locally on CPU with `method="lora"` and the tiny model
`hf-internal-testing/tiny-random-LlamaForCausalLM`. Full reference: [docs/api.md](docs/api.md).

## Architecture

```
Browser tab (React SPA) --HTTPS--> Colab proxy / cloudflared / localhost
      REST /api/*  +  WebSocket /ws
FastAPI app (api/) -- Studio facade (studio.py) -- EventBus (core/events.py)
      security/   data_engine/ + connectors/   training/ + export/   monitoring/
            all inside the notebook kernel process (uvicorn in a daemon thread)
```

Packages talk only through `contracts.py` (Pydantic models), `interfaces.py` (Protocols), `errors.py`
and the EventBus. Heavy ML dependencies are imported lazily, so the API, data engine and UI work
without torch. See [docs/architecture.md](docs/architecture.md) and the ADRs in
[docs/decisions/](docs/decisions/).

## Multi-agent development

The project is built by a team of specialised agents, coordinated by a Chief Architect.

- [`AGENTS.md`](AGENTS.md) lists the roster, who owns which paths, the workflow and the rules.
- `.agents/agents/*.md` holds the agent definitions, and `.agents/skills/*/SKILL.md` the skills.
- `python scripts/sync_agents.py` validates every agent and skill and mirrors them into
  `.claude/agents` / `.claude/skills`. `python scripts/sync_agents.py --check` only validates.
  The check mode was run here: `skills: 10  agents: 10`.
- `tests/unit/test_agent_config.py` fails if an agent references a missing skill.

## Testing

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests        # full suite (~100 s)
.venv/Scripts/python.exe -m pytest -q tests -m "not slow"       # skip tiny-model training / large files
.venv/Scripts/python.exe -m pytest -q tests --run-network       # include Hugging Face Hub tests
cd frontend && npm test -- --run && npm run build               # frontend unit tests + production build
```

Last run (2026-11-14, Windows 11, CPython 3.11, CPU): `574 passed, 9 skipped` (Python) and
`54 passed` (frontend, 6 files).

## Status

Summary of [docs/acceptance.md](docs/acceptance.md), which has 50 criteria:

| Status | Count | Notes |
|---|---|---|
| PASS | 45 | Some of these are only simulated, or PASS here but UNVERIFIED on GPU or a real VM for a sub-part: R3, R5, R6, R9, D6, I8, T8, T10, T11, L6 |
| FAIL | 0 | DEF-QA-01 is fixed |
| UNVERIFIED | 5 | R1 Colab, R2 Kaggle, D1 Hub streaming (network tests are opt-in), T2 QLoRA/Unsloth on CUDA, L7 Colab Drive persistence |

Open limitations: [docs/known-limitations.md](docs/known-limitations.md).

## Documentation

- [Installation](docs/installation.md): local dev, Colab, Kaggle, building a wheel
- [Colab guide](docs/colab.md) and [Kaggle guide](docs/kaggle.md)
- [API reference](docs/api.md): Python `Studio`, REST routes, WebSocket events, error codes
- [Configuration](docs/configuration.md): `AUTOTUNE_*` environment variables, model and training fields
- [Troubleshooting](docs/troubleshooting.md)
- [Known limitations](docs/known-limitations.md)
- [Security review](docs/security-review.md), [acceptance matrix](docs/acceptance.md),
  [progress report](docs/progress-report.md)

## License

Apache-2.0 (as declared in `pyproject.toml`). The repository does not contain a separate `LICENSE`
file yet.
