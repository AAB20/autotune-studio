import { describe, expect, it, vi } from "vitest";
import { bootstrapAuth, loginWithToken, parseAuthFragment } from "../services/auth";
import { mockFetch } from "./helpers";

const ENV = {
  kind: "colab",
  python_version: "3.11.9",
  platform: "Linux",
  is_notebook: true,
  workspace_dir: "/content/autotune",
  allowed_roots: ["/content"],
};

function fakeWindow(hash: string, pathname = "/proxy/7860/") {
  const loc = { hash, pathname, search: "" };
  const replaceState = vi.fn((_s: unknown, _t: string, url: string) => {
    loc.hash = url.slice(url.indexOf("#"));
  });
  return { win: { location: loc, history: { state: null, replaceState } } as unknown as Window, replaceState, loc };
}

describe("parseAuthFragment", () => {
  it.each([
    ["#token=abc", "abc", "/"],
    ["token=abc", "abc", "/"],
    ["#token=abc&x=1", "abc", "/?x=1"],
    ["#/?token=abc", "abc", "/"],
    ["#/monitor?token=a-b_c", "a-b_c", "/monitor"],
    ["#/datasets?foo=1&token=zz", "zz", "/datasets?foo=1"],
    ["#/datasets", null, "/datasets"],
    ["#/", null, "/"],
    ["", null, "/"],
    ["#token=", null, "/"],
  ])("%s -> token=%s route=%s", (hash, token, route) => {
    expect(parseAuthFragment(hash)).toEqual({ token, route });
  });

  it("decodes url-encoded tokens", () => {
    expect(parseAuthFragment("#token=a%2Bb").token).toBe("a+b");
  });
});

describe("bootstrapAuth", () => {
  it("strips the token from the URL, POSTs the session, then verifies via /api/env", async () => {
    const calls: string[] = [];
    const fetchMock = mockFetch({
      "POST /api/auth/session": (init) => {
        calls.push(`session:${String(init?.body)}`);
        return { status: 204 };
      },
      "GET /api/env": () => {
        calls.push("env");
        return { body: ENV };
      },
    });
    const { win, replaceState, loc } = fakeWindow("#token=s3cret");
    const state = await bootstrapAuth(win);

    expect(state).toEqual({ status: "authenticated", env: ENV });
    expect(replaceState).toHaveBeenCalledTimes(1);
    expect(replaceState.mock.calls[0]?.[2]).toBe("/proxy/7860/#/");
    expect(loc.hash).not.toContain("s3cret");
    expect(calls).toEqual([`session:${JSON.stringify({ token: "s3cret" })}`, "env"]);
    const [, init] = fetchMock.mock.calls[0]!;
    expect(init?.credentials).toBe("same-origin");
    expect(init?.method).toBe("POST");
  });

  it("strips the token before the network call even if the session POST fails", async () => {
    mockFetch({
      "POST /api/auth/session": { status: 401, body: { code: "unauthorized", message: "bad token" } },
      "GET /api/env": { status: 401, body: { code: "unauthorized", message: "no session" } },
    });
    const { win, replaceState } = fakeWindow("#/monitor?token=wrong");
    const state = await bootstrapAuth(win);
    expect(replaceState.mock.calls[0]?.[2]).toBe("/proxy/7860/#/monitor");
    expect(state.status).toBe("unauthenticated");
  });

  it("returns unauthenticated (token screen) when there is no token and /api/env is 401", async () => {
    mockFetch({ "GET /api/env": { status: 401, body: { code: "unauthorized", message: "no session" } } });
    const { win, replaceState } = fakeWindow("#/");
    const state = await bootstrapAuth(win);
    expect(state.status).toBe("unauthenticated");
    expect(replaceState).not.toHaveBeenCalled();
  });

  it("reuses an existing cookie session without a token", async () => {
    mockFetch({ "GET /api/env": { body: ENV } });
    const { win } = fakeWindow("");
    expect((await bootstrapAuth(win)).status).toBe("authenticated");
  });

  it("reports server errors separately from auth failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    const { win } = fakeWindow("");
    const state = await bootstrapAuth(win);
    expect(state.status).toBe("error");
  });

  it("loginWithToken maps a rejected token to unauthenticated with a reason", async () => {
    mockFetch({ "POST /api/auth/session": { status: 401, body: { code: "unauthorized", message: "nope" } } });
    const state = await loginWithToken("x");
    expect(state).toEqual({ status: "unauthenticated", reason: "The access token was rejected." });
  });
});
