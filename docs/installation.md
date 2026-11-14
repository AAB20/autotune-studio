# Installation

AutoTune Studio supports Python 3.10 to 3.13 (`requires-python = ">=3.10,<3.14"`). Only 3.11 has
been exercised. There is no PyPI release yet: install from a source checkout, from a git URL or
from a wheel you build yourself.

## Package extras

| Extra | Installs | Needed for |
|---|---|---|
| *(none)* | fastapi, uvicorn, websockets, pydantic, psutil, pyarrow, polars, ijson, httpx, h11 | UI, API, local CSV/JSON/JSONL/Parquet streaming |
| `hf` | datasets, huggingface_hub | Hugging Face Hub datasets |
| `train` | torch, transformers, peft, trl, accelerate | any training or export |
| `gpu` | unsloth, bitsandbytes, nvidia-ml-py | QLoRA, the Unsloth backend, NVML GPU utilisation (CUDA only) |
| `db` | sqlalchemy | reserved for future database connectors (none implemented yet) |
| `dev` | pytest, pytest-asyncio, pytest-timeout, ruff | running the test suite |

Without `train`, everything except training/export still works. `start_training()` then fails with
`dependency_missing` and an install hint.

## Local development (verified: Windows 11, CPython 3.11, CPU)

Requirements: [uv](https://docs.astral.sh/uv/), Node.js (dev box: Node 24 / npm 12) for the UI.

```bash
# 1. Virtual environment
uv venv .venv -p 3.11

# 2. CPU-only torch first. On Windows PyPI torch is CPU anyway; on Linux this avoids the large CUDA build.
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu

# 3. The package (editable) with Hub datasets, training and dev tools
uv pip install -p .venv -e ".[hf,train,dev]"

# 4. Build the browser UI into autotune_studio/web/
cd frontend
npm ci
npm run build
cd ..

# 5. Check it
.venv/Scripts/python.exe -m autotune_studio env
.venv/Scripts/python.exe -m autotune_studio launch            # opens a new browser tab; Ctrl+C stops
```

On Linux/macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe`. The dev box's exact
package versions are in `requirements/verified-dev.txt`. `requirements/dev.txt` installs
`-e .[hf,train,db,dev]`.

How each step was verified here (2026-11-14):

| Step | Evidence |
|---|---|
| 2 | `uv pip install --dry-run torch --index-url https://download.pytorch.org/whl/cpu` on a fresh 3.11 venv resolved `torch==2.15.0+cpu`. The dev venv already has it installed. |
| 3 | `--dry-run` on a fresh venv: "Would install 69 packages". The dev venv already has an editable install. |
| 4 | `npm ci --dry-run` succeeded. `npm run build` wrote `autotune_studio/web/index.html` and `assets/index-*.{js,css}`. |
| 5 | `python -m autotune_studio env` printed the JSON. `python -m autotune_studio launch --no-browser --tunnel local --port 7863` answered `GET /api/health` → `{"status":"ok","version":"0.1.0"}`, and `/api/env` without credentials → 401. |

If you skip step 4 the backend still runs, but `/` shows a "UI not built" page with the build
command. The UI is detected on every request, so building after launch works without a restart.

### CLI

```
autotune-studio env                      # (= python -m autotune_studio env) runtime + hardware JSON
autotune-studio launch [--port N] [--host 127.0.0.1] [--workspace DIR] [--allow DIR]...
                       [--tunnel colab_proxy|cloudflared|local] [--no-browser]
```

`--allow` adds an extra allowed dataset root and can be repeated. `--no-browser` prints the
launch link (which contains the session token) to your terminal instead of opening it. Without
`--port` the server uses 7860, or the next free port up to 7879, or an OS-assigned one. An explicit
`--port` is used as is and must be free. Outside
Colab/Kaggle the default is **local only**: a public tunnel needs `--tunnel cloudflared`.

### Running the tests

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests     # 574 passed, 9 skipped in ~100 s (2026-11-14)
.venv/Scripts/python.exe -m pytest -q tests -m "not slow"
.venv/Scripts/python.exe -m pytest -q tests --run-network    # or AUTOTUNE_RUN_NETWORK=1; needs internet
cd frontend && npm test -- --run                             # 54 passed (6 files)
cd frontend && npm run typecheck
```

There is also an opt-in real-tunnel smoke test that opens a public quick tunnel. It is
UNVERIFIED here (it was not run):
`AUTOTUNE_TUNNEL_SMOKE=1 AUTOTUNE_TUNNEL_DOWNLOAD=1 pytest --run-network tests/unit/runtime/test_tunnel_smoke.py`.

## Build a wheel

```bash
cd frontend && npm run build && cd ..   # the UI must be built first: the wheel packages autotune_studio/web/**
uv build --wheel                        # -> dist/autotune_studio-0.1.0-py3-none-any.whl
```

Verified 2026-11-14:

- The wheel is about 408 KiB (417,390 bytes) with 72 entries. It contains
  `autotune_studio/web/index.html` and `autotune_studio/web/assets/index-*.js|css`.
- It was installed into a throwaway venv with `uv venv <dir> -p 3.11` and
  `uv pip install -p <dir>/Scripts/python.exe dist/autotune_studio-0.1.0-py3-none-any.whl`.
- In that venv, `python -m autotune_studio env` worked. `autotune-studio --help` listed
  `launch` and `env`.
- `Studio().launch(open_browser=False, prefer="local")` served `/api/health` and the UI's `/`
  (HTTP 200). `/api/env` returned 401 without credentials and 200 with Bearer. After `shutdown()`
  the server no longer answered.
- With pip 26.3, `pip install --dry-run "dist/autotune_studio-0.1.0-py3-none-any.whl[hf]"`
  resolved the extras (`datasets`, `huggingface_hub`). The notebooks use this syntax.

**The built UI is committed.** `autotune_studio/web/` (the Vite build output) is tracked in git so
that `pip install git+<url>` ships the browser UI. After changing anything in `frontend/`, run
`cd frontend && npm run build` and commit `autotune_studio/web/`. `tests/unit/test_web_bundle.py`
checks that the committed `index.html` only references assets that exist. The notebooks still
check for the UI after installing.

## Google Colab (UNVERIFIED)

Use [`notebooks/colab.ipynb`](../notebooks/colab.ipynb). In short:

```python
# keep Colab's preinstalled torch; no --force-reinstall
!pip install -q "/content/autotune_studio-0.1.0-py3-none-any.whl[hf,train]"   # after uploading the wheel
# or: !pip install -q "autotune-studio[hf,train] @ git+https://github.com/AAB20/autotune-studio.git"  # private: see notebooks (GITHUB_TOKEN)
!pip install -q unsloth bitsandbytes nvidia-ml-py
```

Unsloth's recommended install command changes often. If the line above fails or breaks torch,
follow https://github.com/unslothai/unsloth. See [colab.md](colab.md).

## Kaggle (UNVERIFIED)

Use [`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb). Internet must be ON in the notebook
settings. Upload the wheel as a Kaggle dataset and add it as an input:

```python
import glob
whl = sorted(glob.glob("/kaggle/input/**/autotune_studio-*.whl", recursive=True))[-1]
!pip install -q "{whl}[hf,train]"
!pip install -q unsloth bitsandbytes nvidia-ml-py
```

See [kaggle.md](kaggle.md).
