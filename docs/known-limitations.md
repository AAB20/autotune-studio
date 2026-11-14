# Known limitations

Consolidated from `docs/acceptance.md`, `docs/progress-report.md`, `docs/security-review.md`,
`docs/e2e-browser-check.md` and the code, as of 0.1.0 (2026-11-14).

## Not yet verified (UNVERIFIED)

Everything below is implemented, but could not be run on the Windows CPU dev box.

| Area | Item | How to verify |
|---|---|---|
| Colab | Install, launch via the kernel proxy, the UI in a new tab, Drive persistence / runtime recycle (R1, L7) | Run `notebooks/colab.ipynb` top to bottom on a T4 runtime |
| Hub push | `PUT /api/hf/credentials` against the real Hub and pushing an export (tested with a faked `HfApi` only) | Enter a write token on the Settings page and export with *Push to Hugging Face Hub* |
| Kaggle | Install, cloudflared tunnel with `publicly_verified=True` (R2) | Run `notebooks/kaggle.ipynb` with Internet ON |
| Notebooks | `notebooks/colab.ipynb` and `notebooks/kaggle.ipynb` **have never been executed**. They are validated only as JSON / nbformat, and their code cells parse | Execute them on each platform |
| GPU | QLoRA 4-bit, the Unsloth backend (T2), real CUDA OOM recovery (T10), VRAM / NVML monitoring (T11, R5), merged export of 4-bit / Unsloth models (T8) | Colab T4 with `method="qlora"`, plus an oversized batch |
| Hub | Hugging Face dataset streaming (D1). The network tests are opt-in | `pytest --run-network tests/unit/data/test_connectors.py` |
| Jupyter | Launch from a real Jupyter kernel and the notebook link / `window.open` behaviour (R3, I1, R9) | `jupyter lab` → `Studio().launch()` |
| Python versions | Only 3.11 was run. 3.10, 3.12 and 3.13 have not been (R6) | `uv venv -p 3.10` / `-p 3.12` and run the suite |
| Proxies | Long sessions through a real Colab proxy / tunnel (keepalive, L6). PSL status of the Colab proxy domain (SEC-03) | Keep the UI open for more than 10 min during training |
| Real tunnel | cloudflared smoke test against Cloudflare (I8). The pinned SHA256 values were not re-checked against a download (SEC-11) | `AUTOTUNE_TUNNEL_SMOKE=1 AUTOTUNE_TUNNEL_DOWNLOAD=1 pytest --run-network tests/unit/runtime/test_tunnel_smoke.py` |
| Scale | Datasets larger than RAM. The largest tested were 150 MB in the suite and 600 MB CSV by hand (D6) | Stream a multi-GB file |
| Deps | `unsloth` / `bitsandbytes` are unpinned and their compatibility with Colab/Kaggle torch is unknown | Notebooks |

## Functional limitations

- **One training job at a time.** A second start returns `409 conflict`. Training uses a single
  process on one GPU (the current CUDA device). On Kaggle "T4 x2", the second GPU is unused.
- **No global shuffle of streamed datasets.** Global shuffling is impossible without
  materialising the data. Shuffling uses a bounded, seeded buffer, and the train/validation split
  is a deterministic hash of each row (ADR-0003).
- **Validation metric is sampled.** `eval_loss` is estimated on at most **256** validation rows per
  checkpoint (`EVAL_MAX_ROWS`). This caps the metric sample, not training data.
- **Unknown-length datasets need `max_steps`.** You can run a full scan first to switch to epochs.
- **CSV memory overhead.** pyarrow's streaming CSV reader adds a roughly fixed 150–235 MiB of RSS
  regardless of file size. Memory stays bounded, but for 100–200 MB CSV files the growth is not
  far below the file size. JSONL adds about 65 MiB.
- **No database connectors yet.** The `db` extra (SQLAlchemy) is reserved. Planned connectors will
  use keyset pagination / server-side cursors, never `OFFSET` (ADR-0001).
- **Uploads persist.** Successfully loaded uploads stay in `<workspace>/uploads/` after shutdown.
  Delete them yourself if needed. Failed uploads are deleted (SEC-12).
- **Non-file multipart fields are not size-capped.** They are drained, not buffered, and need
  authentication (SEC-10).
- **Hugging Face credentials last for the session only.** They are kept in memory (by design) and must
  be entered again on the Settings page after a kernel restart. The UI token is used only for pushing;
  gated downloads use the `HF_TOKEN` notebook secret.

## Networking

- **Quick tunnels have no uptime guarantee.** `trycloudflare.com` tunnels can drop or be
  rate-limited. A restart (`shutdown()` + `launch()`) gives a new URL.
- **The Colab proxy URL is never "publicly verified".** It works only in the browser signed in to
  that Colab session, and the kernel cannot probe it.
- **Long requests through Cloudflare may time out (about 100 s).** A merged 16-bit export
  (`POST /api/export` with `merged_16bit`) of a real model can take longer and return a proxy
  524, even though the export continues on the server. Run `studio.export_model(...)` from the
  notebook instead. Save-now is capped at 90 s for the same reason and returns `409` instead.
- **Slow WebSocket clients may drop events.** Each subscriber queue holds 1000 events. When it is
  full the oldest are dropped, without a `gap` flag until the client reconnects. REST state is
  authoritative.
- **Rate limiting behind a tunnel is global.** All clients share 127.0.0.1, so a hostile client
  can lock the owner out of the token exchange for up to 60 s.

## Security notes (accepted risks)

- **The launch link is a credential while the kernel runs** (SEC-08). Clear notebook outputs
  before sharing or saving a version, and call `studio.shutdown()`.
- **Path checks have a TOCTOU window.** A symlink swapped in between validation and open is not
  detected. This is acceptable for a single-user notebook.
- **Redaction is pattern-based.** Secrets in unusual encodings may not be masked.
- **Binaries are trusted without hashing.** `AUTOTUNE_CLOUDFLARED` and binaries found on `PATH`
  are not hash-checked.

## Packaging

- **No PyPI release.**
- **The built UI must be rebuilt and committed by hand** after frontend changes (`npm run build`);
  `tests/unit/test_web_bundle.py` only checks that the committed bundle is self-consistent.
