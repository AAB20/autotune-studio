# ADR-0002: Exposing the UI from notebook runtimes

Status: accepted (2026-11-14)

## Context
The backend runs inside the notebook kernel. The UI must open in a separate browser tab. Colab and Kaggle have different networking.

## Decision
Adapters tried in order; the first that works is used:
1. **Colab built-in proxy** — `google.colab.output.eval_js("google.colab.kernel.proxyPort(PORT)")` returns an https URL bound to the user's Google session. No third-party service. Reachable only from the browser signed into that Colab session (probe from the kernel is not possible → `publicly_verified=False` with explanatory note).
2. **cloudflared quick tunnel** — official binary from `github.com/cloudflare/cloudflared/releases` (pinned version, SHA256 verified), `cloudflared tunnel --url http://127.0.0.1:PORT --no-autoupdate`, URL parsed from output, then verified with `GET /api/health` through the tunnel. Used for Kaggle (requires notebook Internet enabled) and as Colab fallback.
3. **Local** — `http://127.0.0.1:PORT` (local Jupyter / scripts). Diagnostics explain when no public access exists.

ngrok is not used (requires an account token).

## Consequences
- Quick tunnels have no uptime guarantee; diagnostics say so. Access is protected by the session token (ADR-0005), since anyone with the URL can reach the login endpoint.
- Never claim public reachability without a successful probe.
