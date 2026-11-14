# Running AutoTune Studio on Google Colab

> **UNVERIFIED.** Nothing in this guide has been run on a real Colab VM yet (acceptance R1, L7).
> The behaviour described here is what the code implements. The unit and compatibility tests
> cover it with a *simulated* Colab (a fake `google.colab` module and a fake proxy).

Notebook: [`notebooks/colab.ipynb`](../notebooks/colab.ipynb).

## 1. Runtime and GPU

*Runtime > Change runtime type > T4 GPU* (or A100/L4). Check it with `!nvidia-smi`. Without a GPU:

- `method="qlora"` is rejected with `validation_error`: "QLoRA ... requires a CUDA GPU". Use
  `method="lora"`, but CPU training is only practical for tiny models.
- On a CPU host the studio defaults its training method to `lora`, and to `qlora` when CUDA is
  available.

## 2. Install

See the notebook's install cell and [installation.md](installation.md#google-colab-unverified):

- **Wheel (recommended).** Build it locally (`npm run build` in `frontend/`, then
  `uv build --wheel`), upload it with `google.colab.files.upload()` and run
  `pip install "/content/<wheel>[hf,train]"`. The wheel includes the browser UI.
- **Git URL** (notebook default `https://github.com/AAB20/autotune-studio.git`).
  The repository is public: no token is needed.
  Equivalent: `pip install "autotune-studio[hf,train] @ git+<url>"`. The built UI is committed in
  the repository, so this includes the browser UI too.
- **GPU extras.** `pip install unsloth bitsandbytes nvidia-ml-py`. Do **not** force-reinstall
  torch: Colab's torch matches its CUDA driver. Unsloth's install instructions change often; if
  pip fails or wants to replace torch, follow https://github.com/unslothai/unsloth. Without
  Unsloth the `hf` backend is used. QLoRA then needs `bitsandbytes`.

If pip asks you to restart the session, do it and continue after the install cell.

## 3. Hugging Face token (optional)

The token is needed for gated or private models and datasets. Add a Colab secret named
`HF_TOKEN` (key icon in the sidebar) and grant this notebook access. The studio looks for a token
in this order: env `HF_TOKEN`, env `HUGGING_FACE_HUB_TOKEN`, Colab `userdata`, Kaggle secrets.
Dataset streaming picks it up automatically. The notebook also copies it into `os.environ["HF_TOKEN"]`
so model downloads (transformers / huggingface_hub) see it. The value is registered for log
redaction and never printed or returned by the API.

## 4. Workspace and Google Drive

| What | Default on Colab |
|---|---|
| Workspace (runs, checkpoints, exports, uploads, `sessions.json`) | `/content/autotune` |
| Allowed dataset roots | `/content` plus the workspace plus `AUTOTUNE_ALLOWED_ROOTS` |

Everything under `/content` is lost when the runtime is recycled. If Drive is not mounted,
`detect_environment().notes` warns about this. To persist your work:

```python
from google.colab import drive
drive.mount("/content/drive")
studio = Studio(workspace_dir="/content/drive/MyDrive/autotune")
```

Drive persistence of checkpoints and resuming after a recycle are UNVERIFIED (acceptance L7).
Writing large merged models to Drive can be slow.

## 5. Networking: how the UI reaches your browser

`studio.launch()` starts uvicorn on `127.0.0.1:<port>` (7860 or the next free port) in a daemon
thread, so the cell returns immediately. It then tries these adapters in order:

1. **Colab proxy (default).** It calls
   `google.colab.output.eval_js("google.colab.kernel.proxyPort(PORT)")`, which returns an https
   URL served by Google. The URL is tied to your Google session, so open it **in the same
   browser that runs the notebook**. The kernel cannot probe it (the probe would need your
   browser's cookies), so `publicly_verified` is always `False` on Colab, with a diagnostic
   explaining why.
2. **cloudflared quick tunnel (fallback).** Used if the proxy is unavailable. See
   [kaggle.md](kaggle.md#networking-cloudflared-quick-tunnel) for how it works.
   `publicly_verified=True` only after `GET <tunnel>/api/health` returned 200.
3. **Local.** `http://127.0.0.1:<port>` is reachable only inside the VM, not from your browser.
   The diagnostics say so.

You can force an adapter with `studio.launch(prefer="cloudflared")` (or `"colab_proxy"` /
`"local"`).

The notebook shows an **Open AutoTune Studio** button (`target="_blank"`) and also tries
`window.open`, which popup blockers may stop. The button always works. The link looks like
`<url>/#token=<session token>`. The token lives only in the fragment, which is never sent to
servers or proxies. The SPA exchanges it for an HttpOnly cookie and removes it from the address
bar.

`info = studio.launch()` returns a `LaunchInfo`. Print `info.method`, `info.publicly_verified`,
`info.public_url` and `info.diagnostics`. **Do not print `info.open_url`**: it contains the token.

## 6. Security

- Anyone who has the full link can control the studio while the kernel runs. **Clear all outputs
  (*Edit > Clear all outputs*) before sharing or saving the notebook to GitHub**, and run
  `studio.shutdown()` when you are done. The token dies with the process.
- The WebSocket sends a keepalive every 20 s (`AUTOTUNE_WS_PING_INTERVAL`) so idle proxies do not
  drop it. Behaviour on the real Colab proxy over long sessions is UNVERIFIED (acceptance L6).

## 7. If the UI does not open

The backend keeps running even if every adapter fails. See
[troubleshooting.md](troubleshooting.md). In short:

- Re-display the button: `studio.launch()` again returns the same `LaunchInfo` and shows the link
  again.
- 401 or the token-entry screen: use the newest link. Each `Studio()` has its own token, and the
  cookie is only valid for the process that issued it.
- "Reconnecting" badge: the WebSocket reconnects with backoff and replays missed events. Check
  that the kernel is still alive.
- Use the Python API instead. Every UI action has a `Studio` method ([api.md](api.md)).

## Pushing your model to the Hugging Face Hub

After `studio.launch()`, open **Settings** in the UI and enter your Hugging Face username (or an
organisation you belong to) and an access token with **Write** permission
(https://huggingface.co/settings/tokens). The token is verified with the Hub and kept in the
server's memory for this session only. Then, on **Checkpoints > Export**, switch on
*Push to Hugging Face Hub*, choose a repository name and export. (UNVERIFIED against the real Hub.)
