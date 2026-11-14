# ADR-0007: Hugging Face login in the UI and push-to-Hub export

Status: accepted (2026-11-14)

## Context
Users want to publish fine-tuned models to the Hugging Face Hub from the browser UI, entering their
Hugging Face username and access token after `studio.launch()`, instead of configuring notebook secrets.

## Decision
- `PUT /api/hf/credentials {username, token}` verifies the token with `HfApi.whoami` and checks that the
  username is the token's user or one of its organisations. Credentials are held **in memory only** by the
  `Studio` instance (never written to disk, never returned by any endpoint, registered with log/event
  redaction). `GET` returns `HFCredentialsStatus` (username, verified, write role); `DELETE` forgets them.
- `ExportRequest` gains `push_to_hub`, `hub_repo_name`, `hub_private`. After a successful local export the
  folder is uploaded with `create_repo(exist_ok=True)` + `upload_folder` to `<username>/<repo>`;
  `autotune_state.json` (contains local paths) is not uploaded. `ExportResult` gains `hub_repo_id`, `hub_url`.
- Hub failures map to `errors.HubError` (`hub_error`, 502) with the Hub's message (redacted).
- The UI token is used **only for pushing**. Downloads of gated models still use `HF_TOKEN`
  (env / Colab / Kaggle secret), as chosen by the user.

## Consequences
Credentials are lost when the kernel restarts (by design). Pushing a merged model uploads several GB through
the notebook VM; the request is synchronous, so very large uploads may outlive a tunnel's proxy timeout —
the `export` event still reports completion over the WebSocket.
