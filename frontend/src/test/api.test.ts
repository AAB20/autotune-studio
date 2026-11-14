import { describe, expect, it, vi } from "vitest";
import { ApiRequestError, api, basePath, fieldErrors, normalizeApiError, parseErrorResponse } from "../services/api";
import { mockFetch } from "./helpers";

describe("normalizeApiError", () => {
  it("passes through the contract ApiError shape", () => {
    expect(normalizeApiError(409, { code: "conflict", message: "busy", detail: { job_id: "j" } })).toEqual({
      code: "conflict",
      message: "busy",
      detail: { job_id: "j" },
    });
  });

  it("maps FastAPI {detail: string}", () => {
    expect(normalizeApiError(401, { detail: "Not authenticated" })).toEqual({
      code: "unauthorized",
      message: "Not authenticated",
      detail: null,
    });
  });

  it("maps FastAPI 422 validation arrays", () => {
    const e = normalizeApiError(422, { detail: [{ loc: ["body", "learning_rate"], msg: "must be > 0", type: "x" }] });
    expect(e.code).toBe("validation_error");
    expect(e.message).toBe("must be > 0");
    expect(e.detail).toEqual({ errors: [{ loc: ["body", "learning_rate"], msg: "must be > 0", type: "x" }] });
  });

  it("falls back for text / empty bodies", () => {
    expect(normalizeApiError(502, "Bad Gateway")).toEqual({ code: "http_502", message: "Bad Gateway", detail: null });
    expect(normalizeApiError(500, null, "Internal Server Error").message).toBe("Internal Server Error");
  });
});

describe("parseErrorResponse / request", () => {
  it("parses a JSON error body into ApiRequestError", async () => {
    const err = await parseErrorResponse(
      new Response(JSON.stringify({ code: "forbidden_path", message: "outside allowed roots" }), { status: 403 }),
    );
    expect(err).toBeInstanceOf(ApiRequestError);
    expect(err.status).toBe(403);
    expect(err.code).toBe("forbidden_path");
    expect(err.message).toBe("outside allowed roots");
  });

  it("parses non-JSON error bodies", async () => {
    const err = await parseErrorResponse(new Response("<html>oops</html>", { status: 500, statusText: "ISE" }));
    expect(err.code).toBe("http_500");
    expect(err.message).toContain("oops");
  });

  it("api functions throw ApiRequestError on non-2xx and send same-origin credentials", async () => {
    const f = mockFetch({
      "POST /api/training/start": { status: 409, body: { code: "conflict", message: "A job is already running" } },
    });
    await expect(api.startTraining("ds1")).rejects.toMatchObject({ status: 409, code: "conflict" });
    const [url, init] = f.mock.calls[0]!;
    expect(String(url)).toBe("/api/training/start");
    expect(init?.credentials).toBe("same-origin");
    expect(init?.body).toBe(JSON.stringify({ dataset_id: "ds1" }));
  });

  it("network failures become status 0 network_error", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(api.env()).rejects.toMatchObject({ status: 0, code: "network_error" });
  });

  it("encodes path params and the preview n", async () => {
    const f = mockFetch({ "GET /api/datasets/a%2Fb/preview?n=20": { body: { columns: [], rows: [] } } });
    await api.previewDataset("a/b", 20);
    expect(String(f.mock.calls[0]![0])).toBe("/api/datasets/a%2Fb/preview?n=20");
  });
});

describe("fieldErrors", () => {
  it("extracts per-field messages from pydantic-style errors and max_steps_required", () => {
    const e1 = new ApiRequestError(422, {
      code: "validation_error",
      message: "invalid",
      detail: { errors: [{ loc: ["body", "lora_r"], msg: "too big" }, { loc: ["learning_rate"], msg: "gt 0" }] },
    });
    expect(fieldErrors(e1)).toEqual({ lora_r: "too big", learning_rate: "gt 0" });
    const e2 = new ApiRequestError(422, { code: "max_steps_required", message: "max_steps is required", detail: null });
    expect(fieldErrors(e2)).toEqual({ max_steps: "max_steps is required" });
    expect(fieldErrors(new Error("x"))).toEqual({});
  });
});

describe("basePath", () => {
  it("returns the serving directory", () => {
    expect(basePath("/")).toBe("/");
    expect(basePath("/proxy/7860/")).toBe("/proxy/7860/");
    expect(basePath("/proxy/7860/index.html")).toBe("/proxy/7860/");
    expect(basePath("/proxy/7860")).toBe("/proxy/7860/");
  });
});
