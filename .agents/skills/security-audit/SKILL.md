---
name: security-audit
description: Threat model and review procedure: session-token auth, path allowlist, upload validation, no remote code, secret redaction, tunnel exposure, dependency risks. Load for security/ and for any review.
---
# security-audit

## Purpose
Prevent unauthorised control of a GPU runtime that is reachable through a public tunnel.

## Activation conditions
- Editing `autotune_studio/security/**`; reviewing changes; adding endpoints, file access or dependencies.

## Required inputs
- `docs/security-review.md` (threat model + findings), API route list, ADR-0005 (auth).

## Execution procedure
1. Auth (`security/authentication.py`): `secrets.token_urlsafe(32)` per launch. Client presents it once via `POST /api/auth/session` → HttpOnly, `SameSite=Strict`, `Secure` when served over https, cookie holding an HMAC-derived session value; or `Authorization: Bearer <token>` for scripts. Compare with `hmac.compare_digest`. WebSocket: cookie or `Authorization` header; reject cross-origin `Origin` headers that don't match `Host`.
2. Paths (`security/validation.py`): `Path(p).expanduser().resolve()`; reject unless under an allowed root (`EnvironmentInfo.allowed_roots`); resolution follows symlinks so escapes are caught; extension allowlist per source; reject NUL bytes.
3. Uploads: max size (default 2 GiB, configurable), extension allowlist, streamed to `workspace/uploads/<random>.<ext>` in chunks.
4. Models/datasets: `trust_remote_code=False` hard-coded; no pickle/eval; HF dataset scripts disabled.
5. Secrets: HF token from env `HF_TOKEN` / Colab `userdata` / Kaggle secrets only; never returned by the API; `security.redaction.RedactingFilter` masks `hf_…`, bearer tokens and the session token in all log records.
6. Record review findings with severity + status in `docs/security-review.md`.

## Constraints
- No unauthenticated endpoint except `GET /api/health` (no sensitive data), `POST /api/auth/session`, and static assets.
- Token never in query strings or logs.

## Expected outputs
- Security module, tests, review report.

## Validation checklist
- [ ] 401 for every protected route and WS without credentials.
- [ ] Path traversal tests (`..`, absolute outside roots, symlink escape, NUL byte).
- [ ] Redaction test for `hf_…` tokens and the session token.
- [ ] Cross-origin WS rejected.
