# ADR-0005: Authentication for the browser UI

Status: accepted (2026-11-14)

## Decision
- On launch the studio generates `secrets.token_urlsafe(32)`. The open URL is `<base>/#token=<token>`; fragments are not sent to servers or proxies and do not appear in access logs.
- The SPA posts the token to `POST /api/auth/session`; the server sets an HttpOnly, SameSite=Strict (Secure on https) cookie holding an HMAC of the token, then the SPA strips the fragment from the address bar.
- Scripts may use `Authorization: Bearer <token>`.
- Every `/api/*` route except `GET /api/health` and `POST /api/auth/session`, and `/ws`, require auth. WebSocket also enforces same-origin `Origin`.
- Tokens and HF credentials never appear in logs (redaction filter) or API responses.
