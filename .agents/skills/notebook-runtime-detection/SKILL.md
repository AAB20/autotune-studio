---
name: notebook-runtime-detection
description: Detect Colab / Kaggle / local Jupyter, hardware (GPU, VRAM, RAM, disk, CUDA) and choose a launch adapter (Colab proxy, cloudflared, local). Load for any work in core/environment.py, core/hardware.py or launch/.
---
# notebook-runtime-detection

## Purpose
Report runtime and hardware truthfully and expose the UI in a separate browser tab using a mechanism that actually works in that runtime.

## Activation conditions
- Editing `autotune_studio/core/environment.py`, `core/hardware.py`, `launch/**`.
- Debugging "UI not reachable" reports.

## Required inputs
- `contracts.EnvironmentInfo`, `HardwareInfo`, `LaunchInfo`.
- ADR-0002 (tunnel strategy).

## Execution procedure
1. Environment: Colab if `google.colab` is importable or `COLAB_RELEASE_TAG` env set; Kaggle if `KAGGLE_KERNEL_RUN_TYPE` env set or `/kaggle/working` exists; Jupyter if an IPython kernel is active (`get_ipython().__class__.__name__ == "ZMQInteractiveShell"` or `ipykernel` in sys.modules); else script.
2. Allowed roots / workspace: Colab `/content`; Kaggle `/kaggle/input` (read), `/kaggle/working` (rw), `/kaggle/temp`; local = CWD + `~/.autotune`. Extra roots from `AUTOTUNE_ALLOWED_ROOTS` (os.pathsep-separated).
3. Hardware: psutil (RAM/CPU), `shutil.disk_usage(workspace)`, torch.cuda (lazy import, optional) for GPUs/CUDA/bf16, NVML (`pynvml` from `nvidia-ml-py`) for utilisation when importable. Every probe is wrapped; failures become notes, never exceptions.
4. Launch order: Colab → `google.colab.output.eval_js("google.colab.kernel.proxyPort(PORT)")` (returns a public https URL scoped to the user's browser session); else cloudflared quick tunnel (download pinned release from github.com/cloudflare/cloudflared releases, verify SHA256, start `cloudflared tunnel --url http://127.0.0.1:PORT --no-autoupdate`, parse `https://*.trycloudflare.com` from output within a timeout); else local only.
5. Probe `GET {public_url}/api/health` before setting `publicly_verified=True`. (Colab proxy URLs need the user's browser cookies, so for Colab set `publicly_verified=False` with a note that the URL is accessible from the same browser session.)
6. Kaggle: probe outbound internet (`internet_access`); if disabled, diagnostic: enable "Internet" in the notebook settings panel (requires phone-verified account).

## Constraints
- Never claim the UI is public without a successful probe.
- Session token only in the URL fragment (`#token=`), never query string.
- Keep the backend running if tunnelling fails.

## Expected outputs
- `EnvironmentInfo`, `HardwareInfo`, `LaunchInfo` with actionable diagnostics.

## Validation checklist
- [ ] Unit tests monkeypatch env vars / `sys.modules` for Colab, Kaggle, Jupyter, script.
- [ ] Hardware detection works with torch absent (monkeypatched import failure).
- [ ] Tunnel failure paths tested (binary unavailable / no URL within timeout / checksum mismatch) → method="local", diagnostics non-empty.
