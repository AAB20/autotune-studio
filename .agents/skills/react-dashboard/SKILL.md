---
name: react-dashboard
description: React + TypeScript + Vite + Tailwind + Radix/shadcn-style dashboard conventions, API client, WebSocket hook and page structure. Load for frontend/.
---
# react-dashboard

## Purpose
A professional dark-mode dashboard that drives the backend and shows live training state.

## Activation conditions
- Any change in `frontend/**`.

## Required inputs
- `frontend/src/services/types.ts` (generated from `contracts.py` by `scripts/gen_ts_types.py`), route list in the fastapi-backend skill.

## Execution procedure
1. Stack: Vite, React, TypeScript strict, Tailwind CSS, Radix primitives wrapped shadcn-style in `src/components/ui/`, TanStack Query for REST, Recharts for charts, lucide-react icons, react-router with **hash routing** (works behind Colab proxy paths).
2. Auth bootstrap (`src/services/auth.ts`): read `#token=` from `location.hash`, `POST /api/auth/session {token}` (sets HttpOnly cookie), then `history.replaceState` to strip the fragment. All requests use relative URLs with `credentials: "same-origin"`.
3. `useEvents()` hook (`src/hooks/useEvents.ts`): WebSocket to `/ws?since=<lastSeq>` (ws/wss from `location`), exponential backoff reconnect (0.5 s → 10 s, jitter), connection state `connecting|open|reconnecting|offline`, dispatches events into a small store (zustand or React context).
4. Pages (`src/pages/`): Dashboard (env, GPU, RAM/VRAM, recent sessions), Datasets (source form, upload, schema/columns, sample preview, streaming stats, scan/cancel), Model (HF id, suggested Unsloth models, max seq len, 4-bit), Training (LoRA/QLoRA + hyperparameters, max_steps required hint when cardinality unknown), Monitor (live loss/LR charts, resources, step/ETA, pause/resume/stop), Checkpoints (history, save now, resume, export).
5. Vite `build.outDir = ../autotune_studio/web`, `emptyOutDir: true`, `base: "./"`; dev server proxies `/api` and `/ws` to `127.0.0.1:7860`.

## Constraints
- No hard-coded row limits; previews are labelled "sample".
- Never display or log the token.
- Accessible: labelled inputs, keyboard focus states, sufficient contrast.

## Expected outputs
- Buildable SPA + vitest/testing-library tests.

## Validation checklist
- [ ] `npm run build`, `npm test -- --run`, `npx tsc --noEmit` pass.
- [ ] Reconnect/backoff logic unit-tested.
- [ ] Manually checked in a browser against a live backend.
