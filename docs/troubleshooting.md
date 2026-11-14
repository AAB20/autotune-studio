# Troubleshooting

Start with the diagnostics: `studio.detect_environment().notes`, `studio.detect_hardware().notes`,
`info.diagnostics` from `studio.launch()`, and `studio.training_status().error`. Each entry is meant
to be actionable.

## "QLoRA ... requires a CUDA GPU" / no GPU

`validation_error` with `detail = {"field": "method", "suggestion": "lora"}`. QLoRA quantises the
base model to 4 bits, which needs CUDA. The studio does not silently downgrade.

- On Colab/Kaggle, switch to a GPU runtime (*Runtime > Change runtime type* / *Settings >
  Accelerator*), check with `!nvidia-smi`, and confirm `studio.detect_hardware(refresh=True).cuda_available`.
  If `nvidia-smi` shows a GPU but `cuda_available` is `False`, torch was probably replaced by a CPU
  build during install. Reinstall following the Unsloth/Colab instructions.
- Without a GPU, use `configure_training(method="lora")` and a tiny model. `adamw_8bit` /
  `paged_adamw_8bit` are also rejected without CUDA; use `adamw_torch`.
- `dependency_missing` ("QLoRA needs bitsandbytes or unsloth"): `pip install bitsandbytes`, or
  install Unsloth.

## `max_steps_required` (422)

The dataset is streamed and its length is unknown (`cardinality.rows is None`), so epochs cannot
be converted into steps. Either:

- set `max_steps` (UI: the Training page blocks *Start* and shows "Required: dataset size
  unknown"), or
- run a full scan first (`studio.scan_dataset()` or *Scan* in the UI). The scan counts every row
  without keeping them, and afterwards `num_epochs` works. Both paths were verified locally.

Parquet files and many Hugging Face datasets report their length from metadata and do not need
this.

## 401 / token-entry screen

- The link you opened is from an older launch, or from another `Studio()` instance. Each instance
  has its own token, and its cookie is signed with a key that exists only in that process.
  Re-display the current link with `studio.launch()`, which shows the same link again while the
  server runs.
- Make sure you open the **whole** link, including `#token=...`. Some chat or copy tools drop
  the fragment.
- `429 rate_limited`: more than 20 failed token attempts per minute. Wait for `Retry-After`.
  Behind a tunnel every client shares one IP, so this limit is effectively global.
- Opening a new link in an already-open tab re-runs the token exchange (the `hashchange`
  handler).

## WebSocket shows "Reconnecting"

The UI reconnects forever with exponential backoff and resumes from the last `seq`, so no events
are lost or duplicated (verified: acceptance I7). If it never reconnects:

- the kernel died or was restarted: re-run the launch cell and open the new link;
- `studio.shutdown()` was called;
- the tunnel was dropped (quick tunnels have no uptime guarantee): `studio.shutdown()` then
  `studio.launch()` gives you a new URL.

If the `hello` message reports `gap=true`, older events fell out of the replay buffer. The UI
refetches state over REST. Raise `AUTOTUNE_REPLAY_SIZE` if this happens often.

## Kaggle: Internet is off

The notes contain: "Kaggle: outbound internet appears to be disabled ...". Without internet, pip
installs, Hugging Face downloads and the tunnel all fail, and the tunnel is not attempted. Open
the notebook *Settings* panel and switch **Internet ON**. This requires a phone-verified account.
Then restart the session and run again. Local datasets under `/kaggle/input` still work from the
Python API.

## Tunnel URL not verified

`publicly_verified=False` means one of the following:

- **Colab proxy.** Always `False` by design: the kernel cannot probe a URL that needs your
  browser's Google cookies. Open it in the same browser as the notebook.
- **cloudflared.** "did not answer /api/health within 30s ... NOT verified". The quick tunnel's
  DNS can take a while to propagate. Wait and try the link. If it keeps failing, run
  `studio.shutdown()`, then `studio.launch(tunnel_timeout=60)`.
- **"cloudflared tunnel failed"** (download, checksum or no URL): install cloudflared manually and
  set `AUTOTUNE_CLOUDFLARED` to its path.

The backend keeps running locally in all of these cases, and the Python API works.

## Out of memory during training

With `adaptive_batch_size=True` (the default), a CUDA OOM halves the micro-batch and raises
gradient accumulation so the effective batch is kept. Training then resumes from the last
checkpoint. A `log` warning event describes the change, for example "micro-batch 8->4, grad
accumulation 2->4". This was verified with a simulated OOM only; real GPU OOM is UNVERIFIED.
If it still fails at micro-batch 1 (`out_of_memory`, 507):

- reduce `max_seq_length`;
- use QLoRA / `load_in_4bit=True`, or a smaller model;
- lower `lora_r`;
- lower `save_steps` so less work is repeated after a recovery.

Host RAM: the data engine sizes batches from available RAM. CSV parsing through pyarrow has a
fixed overhead of about 150–235 MiB (see known limitations).

## Corrupt or empty dataset (422)

- `dataset_malformed`: the file cannot be parsed as the declared type (for example a non-Parquet
  file named `.parquet`). A failed upload is deleted again. Individual bad JSONL lines are not an
  error: they are counted in `stats.malformed_rows`, with examples in `malformed_examples`. A file
  in which *every* record is malformed is rejected.
- `dataset_empty`: no rows.
- `forbidden_path` (403): the path is outside the allowed roots. Add it with
  `Studio(allowed_roots=[...])`, `--allow DIR` or `AUTOTUNE_ALLOWED_ROOTS`.
- JSON content in a `.json` file that is actually JSON Lines is detected as JSONL automatically.

## Windows notes

- **Symlinks.** Creating symlinks needs admin rights or Developer Mode. Four security tests that
  create symlinks are skipped on the dev box for this reason. NTFS junction escapes *are* tested
  and blocked. `scripts/sync_agents.py` copies files instead of linking them for the same
  reason. `huggingface_hub` may warn that its cache cannot use symlinks: this is harmless (it uses
  more disk). Enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence it
  (UNVERIFIED here).
- **Encoding.** Run the tests with `PYTHONUTF8=1` (as in the documented command) so files are
  read and written as UTF-8.
- **File locks.** Windows cannot delete files that are still open. The upload route closes
  readers before deleting rejected uploads. If you delete workspace files by hand, call
  `studio.shutdown()` first.
- **cloudflared.** On Windows the binary is `cloudflared-windows-amd64.exe` (amd64 only).

## Training on CPU is very slow

This is expected. The `hf` backend runs in float32 on CPU, and the studio warns: "No CUDA GPU
detected: training on CPU is very slow and only practical for tiny models". Use a GPU runtime for
real models. On CPU, use something like `hf-internal-testing/tiny-random-LlamaForCausalLM` with
`max_steps` in the single digits to check the pipeline. 4 steps took seconds on the dev box.

## The browser shows "UI not built"

The package has no built frontend. The repository ships the built UI, so this only happens if
`autotune_studio/web/` was deleted or emptied (for example by a failed `npm run build`). Build it (`cd frontend && npm ci && npm run build`) or install the wheel. A reload
is enough; you do not need to restart the server.
