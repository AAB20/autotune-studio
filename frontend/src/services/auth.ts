/**
 * Auth bootstrap (ADR-0005).
 *
 * The backend opens `<base>/#token=<token>`. Because the SPA uses hash routing, the fragment can
 * also look like `#/route?token=<token>`. We:
 *   1. extract the token from the fragment and immediately strip it from the address bar
 *      (`history.replaceState`) so it never lingers in history or screenshots,
 *   2. exchange it for an HttpOnly cookie via `POST /api/auth/session {token}`,
 *   3. probe `GET /api/env`; a 401 means "not authenticated" -> the UI shows a token-entry screen.
 *
 * The token is never logged, persisted, or kept in module state.
 */
import { ApiRequestError, api, isApiRequestError } from "./api";
import type { EnvironmentInfo } from "./types";

export interface ParsedFragment {
  /** Token found in the fragment, or null. */
  token: string | null;
  /** Fragment with the token removed, without the leading '#', always a route (starts with "/"). */
  route: string;
}

/**
 * Parse a location hash. Supports `#token=abc`, `#token=abc&x=1`, `#/route?token=abc`,
 * `#/route?a=1&token=abc`, `#/` and empty hashes.
 */
export function parseAuthFragment(hash: string): ParsedFragment {
  const frag = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!frag) return { token: null, route: "/" };

  let routePath: string;
  let query: string;
  if (frag.startsWith("/")) {
    const q = frag.indexOf("?");
    routePath = q === -1 ? frag : frag.slice(0, q);
    query = q === -1 ? "" : frag.slice(q + 1);
  } else if (frag.includes("=")) {
    // Bare parameter fragment such as "token=abc".
    routePath = "/";
    query = frag;
  } else {
    routePath = `/${frag}`;
    query = "";
  }

  const params = new URLSearchParams(query);
  const raw = params.get("token");
  const token = raw && raw.trim() ? raw.trim() : null;
  params.delete("token");
  const rest = params.toString();
  return { token, route: rest ? `${routePath}?${rest}` : routePath };
}

/** Remove any token from the address bar, keeping path, query and hash route. Returns the token. */
export function stripTokenFromLocation(win: Pick<Window, "location" | "history"> = window): string | null {
  const { token, route } = parseAuthFragment(win.location.hash);
  if (token !== null) {
    const url = `${win.location.pathname}${win.location.search}#${route}`;
    win.history.replaceState(win.history.state, "", url);
  }
  return token;
}

export type AuthState =
  | { status: "authenticated"; env: EnvironmentInfo }
  | { status: "unauthenticated"; reason?: string }
  | { status: "error"; reason: string };

/** Probe `GET /api/env` and translate the result into an `AuthState`. */
export async function checkSession(): Promise<AuthState> {
  try {
    const env = await api.env();
    return { status: "authenticated", env };
  } catch (e) {
    if (isApiRequestError(e) && (e.status === 401 || e.status === 403)) return { status: "unauthenticated" };
    return { status: "error", reason: e instanceof Error ? e.message : "Unknown error" };
  }
}

/** Exchange a token for a session cookie, then verify the session. */
export async function loginWithToken(token: string): Promise<AuthState> {
  try {
    await api.createSession(token);
  } catch (e) {
    if (e instanceof ApiRequestError && (e.status === 401 || e.status === 403 || e.status === 422)) {
      return { status: "unauthenticated", reason: "The access token was rejected." };
    }
    return { status: "error", reason: e instanceof Error ? e.message : "Unknown error" };
  }
  return checkSession();
}

/** App start-up: consume a fragment token if present, otherwise reuse an existing cookie session. */
export async function bootstrapAuth(win: Pick<Window, "location" | "history"> = window): Promise<AuthState> {
  const token = stripTokenFromLocation(win);
  if (token) {
    const state = await loginWithToken(token);
    if (state.status !== "unauthenticated") return state;
    // Token rejected (e.g. stale link) -> maybe an older cookie session is still valid.
    const existing = await checkSession();
    return existing.status === "authenticated" ? existing : state;
  }
  return checkSession();
}
