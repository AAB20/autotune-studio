/** Test helpers: route-table fetch mock and a render wrapper with QueryClient + router. */
import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";

export interface MockRoute {
  status?: number;
  body?: unknown;
}

export type RouteTable = Record<string, MockRoute | ((init: RequestInit | undefined) => MockRoute)>;

/**
 * Install a `fetch` mock. Keys are "METHOD /path" (query string ignored unless the key has one).
 * Unknown routes return 404 with an ApiError body.
 */
export function mockFetch(routes: RouteTable) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, "http://localhost/");
    const method = (init?.method ?? "GET").toUpperCase();
    const entry = routes[`${method} ${url.pathname}${url.search}`] ?? routes[`${method} ${url.pathname}`];
    const r = typeof entry === "function" ? entry(init) : entry;
    const status = r?.status ?? (r ? 200 : 404);
    const body = r ? r.body : { code: "not_found", message: `no mock for ${method} ${url.pathname}` };
    return new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export function renderWithProviders(ui: ReactElement, { route = "/" }: { route?: string } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } });
  const utils = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...utils, qc };
}
