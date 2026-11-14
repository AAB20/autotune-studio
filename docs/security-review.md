# AutoTune Studio — Security Review

Owner: security agent. Skill: `.agents/skills/security-audit/SKILL.md`. Related: ADR-0005 (auth).

Status: **implementation phase complete** (security module + unit tests). Application-wide review
(routes, WS, training/model loading, launch/tunnel) happens in the review phase; findings go in the
table at the end.

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

| ID | Severity | Component | Finding | Owner | Status |
|---|---|---|---|---|---|
| | | | | | |

Severity scale: Critical / High / Medium / Low / Info. Status: Open / Fixed (commit/test) / Accepted (reason).
