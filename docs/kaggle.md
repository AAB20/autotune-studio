# Running AutoTune Studio on Kaggle

> **UNVERIFIED.** Nothing in this guide has been run in a real Kaggle kernel yet (acceptance R2).
> The behaviour described here is what the code implements. Tests cover it with a *simulated*
> Kaggle (`KAGGLE_KERNEL_RUN_TYPE` set, a fake tunnel).

Notebook: [`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb).

## 1. Notebook settings (right-hand *Settings* panel)

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 x2** or **GPU P100** | QLoRA / Unsloth need CUDA. The studio trains on **one** GPU (the current CUDA device). |
| Internet | **ON** | pip, the Hugging Face Hub and the cloudflared tunnel all need outbound internet. The toggle is only available on a **phone-verified** Kaggle account (*Settings > Phone verification* on your Kaggle profile). |
| Session | interactive | A *Save & Run All* (batch) run has no browser session. The studio adds a note when `KAGGLE_KERNEL_RUN_TYPE=batch`. |

Changing Internet or the accelerator restarts the session. Re-run from the top.

## 2. Install

- **Wheel as a Kaggle dataset (recommended).** Build the wheel locally (see
  [installation.md](installation.md#build-a-wheel)). Upload it via *Datasets > New Dataset*, then
  *Add Input* in the notebook. The install cell finds `/kaggle/input/**/autotune_studio-*.whl` and runs
  `pip install "<wheel>[hf,train]"`. The wheel includes the browser UI. Its dependencies still
  come from PyPI, so Internet must be ON.
- **Git URL** (notebook default `https://github.com/AAB20/autotune-studio.git`, includes the built UI).
  The repository is public: no token is needed.   Internet must be ON.
- **GPU extras.** `pip install unsloth bitsandbytes nvidia-ml-py`. Keep Kaggle's torch; do not
  force-reinstall it. If Unsloth fails to install, follow https://github.com/unslothai/unsloth.

## 3. Hugging Face token (optional)

*Add-ons > Secrets*: create `HF_TOKEN` and attach it to the notebook. The studio reads it through
`kaggle_secrets.UserSecretsClient` when env `HF_TOKEN` is not set. The notebook also exports it to
`os.environ` so model downloads see it. The token is never printed.

## 4. Paths

| What | Kaggle default |
|---|---|
| Workspace | `/kaggle/working/autotune`. It is kept with a saved notebook version (about 20 GB limit). |
| Allowed dataset roots | `/kaggle/input` (read-only), `/kaggle/working`, `/kaggle/temp`, the workspace, `AUTOTUNE_ALLOWED_ROOTS` |

Attached Kaggle datasets can be loaded by path, for example
`studio.load_dataset("csv", path="/kaggle/input/my-data/train.csv")`.

## Networking: cloudflared quick tunnel

Kaggle has no port proxy, so `studio.launch()` uses a Cloudflare quick tunnel:

1. On Kaggle, `Studio()` probes outbound HTTPS (`https://huggingface.co`). If the probe fails,
   `env.internet_access` is `False` and the notes tell you to switch Internet ON. The tunnel is
   then **not attempted**.
2. Binary: `AUTOTUNE_CLOUDFLARED` (an explicit path) if set, otherwise `cloudflared` on `PATH`,
   otherwise a download of the pinned release `2026.11.0` from github.com/cloudflare/cloudflared.
   A downloaded binary is SHA256-verified and cached in `~/.cache/autotune_studio/cloudflared`
   (or `$AUTOTUNE_CACHE_DIR/cloudflared`).
3. The studio runs `cloudflared tunnel --url http://127.0.0.1:PORT --no-autoupdate` and waits up to
   30 s for a `https://*.trycloudflare.com` URL.
4. It then polls `GET <url>/api/health` for up to 30 s. `publicly_verified=True` only if that
   returned 200.

Quick tunnels have **no uptime guarantee**. Anyone with the URL can reach the login endpoint, but
every API and WebSocket call needs the session token from the `#token=` link.

### If the tunnel fails

The **backend keeps running** in every failure case. Read `info.diagnostics`:

| Diagnostic | Meaning / fix |
|---|---|
| "outbound internet appears to be disabled ..." | Turn Internet ON (phone-verified account), restart, re-run. |
| "cloudflared tunnel failed: ..." (download/checksum/not found) | The download was blocked or corrupted. Install cloudflared yourself and set `AUTOTUNE_CLOUDFLARED=/path/to/cloudflared`, then `studio.shutdown()` and `studio.launch()` again. |
| "did not publish a trycloudflare.com URL within 30s" | Cloudflare is slow or rate-limiting. Retry with `studio.launch(tunnel_timeout=60)` after `studio.shutdown()`. |
| "Tunnel URL ... did not answer /api/health within 30s ... NOT verified" | DNS for the new hostname may still be propagating. Wait a minute and open the link anyway; it may start working. |
| "... only reachable from inside the kaggle VM" | No public URL exists. Use the Python API from the notebook ([api.md](api.md)). |

Large merged exports through the UI can exceed Cloudflare's roughly 100 s response timeout. Run
`studio.export_model(format="merged_16bit")` from Python instead
([known-limitations.md](known-limitations.md)).

## Security

The launch link contains the session token. **Clear all outputs before *Save Version***, because a
saved version stores outputs, and before sharing the notebook. Call `studio.shutdown()` when you
are done.

## Pushing your model to the Hugging Face Hub

After `studio.launch()`, open **Settings** in the UI and enter your Hugging Face username (or an
organisation you belong to) and an access token with **Write** permission
(https://huggingface.co/settings/tokens). The token is verified with the Hub and kept in the
server's memory for this session only. Then, on **Checkpoints > Export**, switch on
*Push to Hugging Face Hub*, choose a repository name and export. (UNVERIFIED against the real Hub.)
