# AutoTune Studio — Security Review

Owner: security agent. Skill: `.agents/skills/security-audit/SKILL.md`. Related: ADR-0005 (auth).

Status: **review phase complete (2026-11-14)**. Application-wide review of `api/`, `studio.py`,
`core/`, `launch/`, `training/`, `export/`, `connectors/`, `data_engine/`, `frontend/src/` and the
dependency manifests. Findings: section 4. Dependency risk: section 5. Regression tests:
`tests/unit/security/test_review_findings.py`. Tests for open findings are `xfail(strict=True)`:
once the proposed fix lands they XPASS, which fails the suite, so the marker has to be removed then.

## 1. Threat model

### Assets
| Asset | Why it matters |
|---|---|
| GPU runtime / notebook kernel | Whoever controls the API controls a process that can read the user's Drive / Kaggle inputs, spend GPU quota and (if code execution were reachable) run arbitrary code in the user's account. |
| Hugging Face token | Grants read (and often write) access to the user's private models/datasets on the Hub. |
| Session token / session cookie | Equivalent to full control of the studio for the life of the process. |
| User data | Datasets, checkpoints and exported adapters/merged models on local disk / mounted Drive. |

### Actors
| Actor | Capability |
|---|---|
| Anyone who learns the tunnel URL | A `trycloudflare.com` / Colab proxy URL is reachable from the public internet. URLs leak via screenshots, shared notebooks, browser history, referrers. **Assume the URL is public; the token is the only barrier.** |
| Malicious web page the user visits | Can make the browser send cross-site requests / open WebSockets to the studio origin (CSRF, cross-site WS hijacking). |
| Malicious dataset / model publisher | Supplies files the studio parses (CSV/JSON/Parquet, model repos with custom code, pickles). |
| Co-tenant on a shared host | Can read logs / process listings in some setups. |

### Entry points
| Entry point | Notes |
|---|---|
| HTTP `/api/*` | All routes require auth except `GET /api/health` (no sensitive data) and `POST /api/auth/session`. Static SPA assets are public. |
| WebSocket `/ws` | Event stream (metrics, logs, status). Requires auth + same-origin `Origin`. |
| Uploads | Dataset files streamed to `workspace/uploads/`. |
| Dataset ids / paths | `DatasetSpec.name` (HF id) and `DatasetSpec.path` (local path) from the UI. |
| Model ids / paths | `ModelConfig.model_name` (HF repo id or local directory). |
| Export output dirs | `ExportRequest.output_dir`, `TrainingConfig.output_dir`. |
| Logs | Notebook output, uvicorn access/error logs, WS `log` events. |

## 2. Controls implemented (`autotune_studio/security/`)

| Control | Implementation | Tests (`tests/unit/security/`) |
|---|---|---|
| Per-launch random token | `SessionAuth()` → `secrets.token_urlsafe(32)`; `repr()` masked | `test_generated_token_is_random_and_long` |
| Constant-time verification | `hmac.compare_digest` on UTF-8 bytes (never raises on non-ASCII); 512-char cap | `test_verify_uses_constant_time_compare`, `test_verify_token_accepts_exact_and_rejects_others` |
| Cookie ≠ token | Cookie value = `HMAC-SHA256(per-process random key, token)`; raw token rejected as cookie | `test_cookie_is_hmac_not_token_and_verifies`, `test_cookie_key_is_per_instance` |
| Cookie flags | `cookie_params(secure)` → HttpOnly, SameSite=Strict, Secure (on https), path=/ , session lifetime | `test_cookie_params` |
| Bearer / cookie auth | `authenticate_headers`, `authenticate_connection`, `require_auth_dependency` (raises `Unauthorized` → 401) | `test_header_and_cookie_auth_matrix`, `test_starlette_dependency_401_and_ok` |
| Cross-site WS hijacking | `check_ws_origin`: missing Origin allowed (non-browser, still authenticated), `null`/unparsable rejected, host[:port] must match `Host` or an explicit allowlist entry | `test_check_ws_origin[*]` (21 cases), `test_ws_connection_rejected_cross_origin_and_unauthenticated` |
| Brute-force limit | `FailedAuthLimiter(max_per_minute=20)` per client IP, sliding window, bounded keys; `exchange_token` checks the limit *before* verifying | `test_limiter_*`, `test_exchange_token_flow` |
| Path allowlist | `PathPolicy.resolve`: NUL reject, `expanduser`, `resolve()` (follows symlinks/junctions), containment via `is_relative_to` on `normcase`d paths, containment checked before existence (no oracle), NTFS ADS `:` rejected, relative paths anchored at the first root | `test_dotdot_traversal_rejected`, `test_absolute_outside_root_rejected`, `test_sibling_prefix_is_not_inside`, `test_nul_byte_rejected`, `test_windows_junction_escape_rejected`, `test_symlink_*` |
| Extension allowlist | `ALLOWED_EXTENSIONS` per source; directories only for parquet | `test_extension_allowlist`, `test_directories_only_for_parquet` |
| Upload validation | `sanitize_upload_filename` (basename only, safe charset, ext allowlist); `save_upload_stream` / `asave_upload_stream`: random 128-bit name, exclusive create (`xb`), streamed chunks, `UploadLimits(max_bytes=2 GiB)`, partial file deleted on any failure, empty uploads rejected | `test_upload_*`, `test_sanitize_*` |
| Model id validation | `validate_model_name`: HF `owner/name` regex, no `..`, no URLs/schemes; local dirs only via `PathPolicy` | `test_valid_repo_ids`, `test_invalid_model_names`, `test_local_model_dir_with_policy` |
| Log redaction | `redact()` masks `hf_…`, `Bearer …`, `token=`/`access_token=`/`api_key=` params (incl. `%3D`), `autotune_session=` cookies and registered secrets. `RedactingFilter` redacts msg, args (keeps tuple shape for uvicorn's AccessFormatter), tracebacks, stack info. `install_redaction` attaches to root handlers + `uvicorn*` loggers and wraps the LogRecord factory (covers handlers added later) | `test_redact_patterns`, `test_filter_redacts_msg_args_and_exception`, `test_install_redaction_*`, `test_uvicorn_access_style_args_keep_tuple_shape` |
| Auto-registration of secrets | `SessionAuth` registers token + cookie value; `get_hf_token` registers the HF token | `test_session_auth_registers_token_and_cookie`, `test_hf_token_from_env_and_registered` |
| CSRF defence-in-depth (review, SEC-03) | `authenticate_connection` / `is_same_origin_request`: cookie-authenticated unsafe HTTP methods must carry an `Origin` matching `Host` or a launch origin; without `Origin`, `Sec-Fetch-Site: cross-site/same-site` is rejected. Bearer requests are exempt (not ambient) | `test_sec03_*` |
| HF dataset id validation (review, SEC-01) | `validate_hf_dataset_name`: repo-id regex, no paths/URLs, rejects `datasets` packaged builder names (`csv`, `json`, `text`, ...) and names that exist on the local filesystem | `test_sec01_validate_hf_dataset_name_unit` |
| Model id vs cwd (review, SEC-05) | `validate_model_name`: a repo-id-shaped name that exists relative to the cwd is pinned to its absolute path through `PathPolicy`, or refused | `test_sec05_*` |
| Event payload redaction helper (review, SEC-04) | `redact_data()` recursively redacts strings in JSON-like payloads | `test_sec04_redact_data_unit` |
| HF token sources | env `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → Colab `userdata` → Kaggle `UserSecretsClient`; lazy imports, errors swallowed, value never logged | `test_hf_token_*` |

### Controls owned elsewhere (verify in review phase)
- `trust_remote_code=False` hard-coded — `contracts.ModelConfig` (`Literal[False]`), training loaders.
- No pickle / `eval`; HF dataset loading scripts disabled — connectors / training.
- Token only in URL fragment, never query string — launch.
- Every `/api/*` route (except health + auth/session) and `/ws` wired to the auth helpers — api.
- `POST /api/auth/session` uses `exchange_token` with the real client IP — api.
- `install_redaction()` called at startup (and after uvicorn configures logging) — api/launch.
- HF token never in API responses — api.

## 3. Known residual risks / design notes
1. **TOCTOU on paths**: `PathPolicy.resolve` validates, the caller opens later; a symlink swapped in between is not detected. Acceptable for a single-user notebook; mitigation would be `O_NOFOLLOW`/handle-based checks.
2. **Missing `Origin` allowed on WS**: by design (CLI/script clients); such clients must still present a token or cookie, and browsers always send `Origin` on WS.
3. **Proxies rewriting `Host`** (Colab proxy): same-origin check needs the public origin in `extra_allowed`/`extra_origins`; otherwise browser WS connections are rejected (fail closed).
4. **Rate limit keyed by client IP**: behind cloudflared/Colab proxy all clients may share one IP (the proxy). The limit then applies globally — fails safe (locks out), but a hostile client can block the owner for ≤60 s. Token entropy (256 bit) makes brute force infeasible regardless.
5. **Redaction is pattern-based**: secrets in unforeseen encodings (base64 of a token, split across log args) are not caught. Registered literal secrets are matched exactly.
6. **Registered secrets < 8 chars are ignored** to avoid destroying ordinary log text.
7. `hf_` tokens are matched with ≥20 alphanumerics after the prefix (real tokens: 34).

## 4. Findings (review phase)

Method: code review of every module listed in the status line, plus live attacks against the real
app through Starlette `TestClient`. Attacks tried: path traversal in the dataset path, upload
filename, static path, model name and output dirs; symlink and NTFS junction escapes; WS with no
credentials and from a cross origin; CSRF with cookie auth; oversized and malformed multipart; a
huge preview `n`; XSS payloads in dataset values; tokens in the query string; route-table
enumeration for unauthenticated access. Evidence ids are tests in
`tests/unit/security/test_review_findings.py`.

Severity scale: Critical / High / Medium / Low / Info. Status: Open / Fixed (test) / Accepted (reason).

| ID | Severity | Component | Finding | Owner | Status | Evidence |
|---|---|---|---|---|---|---|
| SEC-01 | **High** | `studio.py` `load_dataset_spec` → `connectors/huggingface.py` | `source="huggingface"` bypasses `PathPolicy`. `DatasetSpec.name` goes straight to `datasets.load_dataset(name)`. That call (a) loads any **local directory or file** that exists, whether the path is absolute or relative to the cwd. (b) For packaged builder names (`csv`, `text`, `json`, `parquet`, ...) without `data_files`, it reads **every matching file under the process cwd, recursively**; `text` reads any file as lines. Live attack: `POST /api/datasets {"source":"huggingface","name":"<dir outside roots>"}` → 200, and the preview returned `admin,hunter2`. | chief-architect | **Open**. Helper `security.validate_hf_dataset_name` is added; the call site is missing | **Fixed** (Architect, 2026-11-14): `Studio.load_dataset_spec` validates HF names with `validate_hf_dataset_name`. |
| SEC-02 | Medium | `pyproject.toml` floors | The floors allow known-vulnerable versions. `torch>=2.2`: CVE-2025-32434 (RCE in `torch.load(weights_only=True)` for torch < 2.6), reachable through `.bin` weights of an attacker-chosen Hub model with older transformers, which `transformers>=4.43` allows. `datasets>=2.19`: 2.19 runs Hub dataset loading scripts with only a warning when `trust_remote_code` is not passed, and the connector does not pass it. `h11` (CVE-2025-43859) is only transitive and has no constraint. The verified env (torch 2.15, transformers 5.23, datasets 5.2, h11 0.16) is not affected. | chief-architect | **Open** | **Fixed** (Architect, 2026-11-14): floors raised: torch>=2.6, transformers>=4.50, datasets>=3.0, h11>=0.16; datasets<4 gets trust_remote_code=False. |
| SEC-03 | Low | `security/authentication.py` (HTTP auth) | Cookie-authenticated state-changing requests were accepted from any `Origin`; only `SameSite=Strict` protected them. Live: `POST /api/training/stop` and a multipart upload with `Origin: https://evil.example` plus the cookie → 200. This is exploitable only same-site, for example from a sibling subdomain on a parent domain that is not on the Public Suffix List, or from browsers that ignore SameSite. `trycloudflare.com` **is** on the PSL (checked 2026-11-14), so sibling quick tunnels count as cross-site. The PSL status of the Colab proxy domain is UNVERIFIED. | security | **Fixed**: `Origin` / `Sec-Fetch-Site` check for cookie auth on unsafe methods; Bearer exempt | `test_sec03_cookie_csrf_cross_origin_rejected[*]`, `test_sec03_sec_fetch_site_without_origin`, `test_sec03_legit_requests_still_work`, `test_sec03_is_same_origin_request_unit` |
| SEC-04 | Low | `core/events.py`, `core/lifecycle.py` | Log redaction only covers `logging` records. Exception text is unredacted wherever it is published outside logging: events (`error`/`log` → WS + replay buffer), `TrainingStatus.error` (REST) and `sessions.json`. Live: an engine raising `...?token=hf_… hf_…` showed the token in `/api/training/status` and in the WS replay. | chief-architect | **Open**. Helper `security.redact_data` is added | **Fixed** (Architect, 2026-11-14): `EventBus.publish` applies `redact_data`; lifecycle errors pass through `redact`. |
| SEC-05 | Low | `security/validation.py` `validate_model_name` | A repo-id-shaped name (`mymodel`, `a/b`) that exists as a directory relative to the cwd is loaded from disk by transformers/Unsloth, because their `os.path.isdir` check runs before the Hub lookup. This bypassed `PathPolicy`. | security | **Fixed**: the name is pinned to its absolute path through `PathPolicy`, or refused | `test_sec05_repo_id_shaped_name_cannot_select_local_dir_outside_roots` |
| SEC-06 | Medium | `export/model_export.py` L128-130 | `tmp = dest.with_name(dest.name + ".partial")`, and if that directory already exists it is `shutil.rmtree`'d. A user directory named `<output_dir>.partial` is silently deleted (data loss). `tmp` is also never checked against the allowed roots: when `output_dir` is a root, `tmp` is its sibling outside the roots. | llm-training | **Open** | **Fixed** (Architect, 2026-11-14): unique `.<name>.<random>.partial` temp dir; nothing pre-existing is deleted. |
| SEC-07 | Low | `connectors/huggingface.py` L75 | The HF token is read only from env `HF_TOKEN`. `security.get_hf_token()` is never called, so the Colab `userdata` and Kaggle secrets sources are unused and the token is never registered as a literal secret. It is masked only by the `hf_` pattern. | data-engineering | **Open** | **Fixed** (Architect, 2026-11-14): HF connector uses `get_hf_token()` (env, Colab userdata, Kaggle secrets). |
| SEC-08 | Info | `launch/browser.py`, `launch/adapters.py` | The open URL (tunnel URL + `#token=`) is written into notebook output as a link and a `window.open` script. Shared `.ipynb` outputs, or a Kaggle "save version" of an interactive session, expose a working credential while the kernel runs. | runtime-compatibility / documentation | **Accepted risk**: the token dies with the process and the fragment never reaches servers or logs. Docs should tell users to clear output before sharing and to call `studio.shutdown()` | review |
| SEC-09 | Info | `export/model_export.py` L105 | Merged Unsloth export: `FastLanguageModel.from_pretrained` does not pass `trust_remote_code=False` explicitly. Unsloth's default is False. | llm-training | **Open** (hardening) | **Fixed** (Architect, 2026-11-14): `trust_remote_code=False` passed explicitly. |
| SEC-10 | Info | `api/multipart.py` | Non-`file` parts are drained with no byte cap, and with chunked encoding the Content-Length pre-check does not apply. The impact is bandwidth/CPU only and needs authentication; the `file` part itself is capped. | backend-engineering | **Accepted risk** | `test_malicious_multipart_is_client_error[*]`, `test_upload_oversized_rejected_and_cleaned` |
| SEC-11 | Info | `launch/cloudflared.py` | The pinned version is SHA256-verified on download and re-verified from cache on every use, and tgz extraction cannot traverse. A non-pinned version (only reachable through an explicit `version=`) takes its checksum from the GitHub API, the same trust domain as the binary. `AUTOTUNE_CLOUDFLARED` and `PATH` binaries are trusted without hashing, by design. | runtime-compatibility | **Accepted**. The pinned hash values themselves were not re-checked against a download (UNVERIFIED) | review |
| SEC-12 | Info | `api/routes/datasets.py` | Uploaded files that loaded successfully stay in `<workspace>/uploads/` after shutdown; failed loads are deleted. | backend-engineering | **Accepted**: user data, local to the workspace | review |

### Verified controls (no finding)
| Check | Result | Evidence |
|---|---|---|
| Unauthenticated surface | Every `/api` route except `GET /api/health` and `POST /api/auth/session` returns 401 + `WWW-Authenticate: Bearer` (enumerated from the router tables). There is no `/docs`, `/redoc` or `/openapi.json` | `test_every_api_route_requires_auth`, `test_no_interactive_docs_or_openapi` |
| Token handling | The token never appears in responses (`/api/launch` strips `open_url`). Query-string tokens are not credentials, and the raw token is rejected as a cookie. The cookie is an HMAC (HttpOnly, SameSite=Strict, Secure on https, session lifetime). The SPA strips the fragment with `history.replaceState` and never stores the token; `localStorage` holds only a dataset id. Requests use `credentials: "same-origin"` | `test_token_never_in_query_or_responses`, `test_session_cookie_flags` |
| WebSocket | Closed with 1008 when there are no credentials, when the Origin is cross-origin, `null` or a suffix spoof, and when the raw token is sent as a cookie. Opens with a same-origin cookie or with Bearer | `test_websocket_auth_and_origin_matrix` |
| Dataset paths | `..`, absolute paths outside the roots, NUL, NTFS ADS and an NTFS junction escape all return 403/422 through the API | `test_file_source_paths_confined`, `test_junction_escape_via_api`. The symlink variant was skipped: no symlink privilege on this box |
| Uploads | Traversal, RFC 2231, device-name and ADS filenames are stored as `uploads/<32 hex>.<ext>`. The extension allowlist holds, the size limit holds with and without Content-Length, malformed multipart returns 422, and no partial files are left | `test_upload_*`, `test_malicious_multipart_is_client_error[*]` |
| Static serving | Encoded `..`, backslashes and absolute drive paths return only the SPA shell | `test_static_traversal_serves_spa_only` |
| XSS via dataset preview | The API returns values as JSON (`nosniff`, `no-store`). The SPA renders cells, column names and log lines as React text. `frontend/src` contains no `dangerouslySetInnerHTML`, `innerHTML` or `eval`. The HTML shell sends CSP `script-src 'self'` | `test_xss_payload_in_dataset_is_inert_json`, `test_frontend_has_no_raw_html_sinks` |
| Preview bounds | `n` is limited to 0..1000; anything else returns 422 | `test_preview_n_is_bounded` |
| Remote code | `ModelConfig.trust_remote_code` is `Literal[False]`, and PUT with `true` returns 422. Every `from_pretrained` passes `trust_remote_code=False`. The package contains no `pickle`/`eval`/`exec`/`yaml.load`/`shell=True`. Parquet is read with pyarrow ≥ 15 (CVE-2023-47248 fixed) | `test_trust_remote_code_cannot_be_enabled`, grep review |
| Output dirs | `TrainingConfig.output_dir` and `ExportRequest.output_dir` are resolved through `PathPolicy` when used. No checkpoint rotation (`save_total_limit=None`) | `test_output_dirs_confined` |
| Unauthorized training commands | All `/api/training/*`, `/api/checkpoints*` and `/api/export` routes are protected | `test_every_api_route_requires_auth` |
| Tunnel | The server binds 127.0.0.1 and the tunnel targets 127.0.0.1. A public tunnel is opt-in outside Colab/Kaggle, and `publicly_verified` is set only after a health probe. The tunnel is stopped by `shutdown()` and `atexit`. DNS rebinding gains nothing: the cookie is host-bound and the token is unknown | review |

### Proposed fixes for open findings (code outside the security agent's paths)
- **SEC-01**, `studio.py` `load_dataset_spec` (L199): for `DatasetSource.HUGGINGFACE` add
  `spec = spec.model_copy(update={"name": validate_hf_dataset_name(spec.name)})`, with the import
  `from .security import validate_hf_dataset_name`. The file-source branch stays unchanged.
- **SEC-02**, `pyproject.toml`: `datasets>=3.0`, `torch>=2.6`, `transformers>=4.50`, and add
  `h11>=0.16` to the core deps. In `connectors/huggingface.py::_load`, pass
  `trust_remote_code=False` when the datasets major version is < 4.
- **SEC-04**: in `core/events.py` L35 use `data=redact_data(data or {})`. In `core/lifecycle.py`
  L296/L299 use `error = redact(...)`, so that `TrainingStatus.error` and `sessions.json` are
  covered too.
- **SEC-06**, `export/model_export.py` L128-131: never delete a directory that already exists.
  Instead use `tmp = dest.parent / f".{dest.name}.{secrets.token_hex(6)}.partial"`, then
  `dest.parent.mkdir(parents=True, exist_ok=True)` and `tmp.mkdir()` (exist_ok=False).
- **SEC-07**, `connectors/huggingface.py` L75: `token = get_hf_token()` (from `..security`).
- **SEC-09**, `export/model_export.py` L105: add `trust_remote_code=False`.

## 5. Dependency risk

**Frontend** (2026-11-14, Node 24 / npm 12): `npm audit --omit=dev` returned `found 0 vulnerabilities`,
and `npm audit` (including dev) also returned `found 0 vulnerabilities`. The runtime deps are
UI-only, and no HTML/markdown renderer is used for dataset content.

**Python**: `pip-audit` is not installed and was deliberately not installed, so this is a
qualitative review of `requirements/verified-dev.txt`.
- The pinned versions have no known issues relevant to this app as of the review date:
  - fastapi 0.162.7 / starlette 1.10.0: the multipart CVE-2024-47874 does not apply, since the app
    uses its own parser and no `python-multipart`.
  - h11 0.16.0 and jinja2 3.1.6: their CVEs are fixed.
  - torch 2.15: CVE-2025-32434 is fixed (fixed in 2.6).
  - pyarrow 26: CVE-2023-47248 is fixed.
  - transformers 5.23 and datasets 5.2: no dataset loading scripts.
  - uvicorn 0.54 and httpx 0.28.1: no relevant issues.
- The risk is in the **floors**, not the pins (SEC-02). Colab and Kaggle install
  `autotune-studio[hf,train,gpu]` over their preinstalled torch/transformers, so the floors decide
  what actually runs there.
- `unsloth` and `bitsandbytes` are unpinned: UNVERIFIED, since there is no GPU env here.
