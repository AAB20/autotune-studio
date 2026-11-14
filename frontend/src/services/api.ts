/**
 * Typed REST client for the AutoTune Studio backend.
 *
 * - All URLs are resolved against the directory the SPA is served from (works behind
 *   Colab / Jupyter proxy paths such as `/proxy/7860/`).
 * - Cookies carry auth (`credentials: "same-origin"`); the session token is never stored here.
 * - Non-2xx responses are parsed into an `ApiError` and thrown as `ApiRequestError`.
 */
import type {
  ApiError,
  CheckpointInfo,
  DatasetInfo,
  DatasetPreview,
  DatasetSpec,
  EnvironmentInfo,
  ExportRequest,
  ExportResult,
  HardwareInfo,
  MetricPoint,
  ModelConfig,
  ResourceSnapshot,
  SessionSummary,
  StreamStats,
  TrainingConfig,
  TrainingStatus,
} from "./types";

/** Error thrown for every failed request. `status` is 0 for network failures. */
export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown> | null;
  readonly apiError: ApiError;

  constructor(status: number, apiError: ApiError) {
    super(apiError.message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = apiError.code;
    this.detail = apiError.detail ?? null;
    this.apiError = apiError;
  }
}

export const isApiRequestError = (e: unknown): e is ApiRequestError => e instanceof ApiRequestError;

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Normalise an error body into `ApiError`. Accepts:
 *  - the contract shape `{code, message, detail?}`
 *  - FastAPI defaults `{detail: "msg"}` and `{detail: [{loc, msg, type}]}` (422)
 *  - anything else (falls back to `http_<status>`).
 */
export function normalizeApiError(status: number, body: unknown, statusText = ""): ApiError {
  const fallbackCode = status === 401 ? "unauthorized" : `http_${status}`;
  if (isRecord(body)) {
    if (typeof body.code === "string" && typeof body.message === "string") {
      const d = body.detail;
      return { code: body.code, message: body.message, detail: isRecord(d) ? d : d == null ? null : { value: d } };
    }
    const d = body.detail;
    if (typeof d === "string") return { code: fallbackCode, message: d, detail: null };
    if (Array.isArray(d)) {
      const first = d.find(isRecord);
      const msg = first && typeof first.msg === "string" ? first.msg : "Validation failed";
      return { code: "validation_error", message: msg, detail: { errors: d } };
    }
  }
  if (typeof body === "string" && body.trim()) {
    return { code: fallbackCode, message: body.trim().slice(0, 300), detail: null };
  }
  return { code: fallbackCode, message: statusText || `Request failed (HTTP ${status})`, detail: null };
}

/** Read a failed `Response` body (JSON or text) and build an `ApiRequestError`. */
export async function parseErrorResponse(res: Response): Promise<ApiRequestError> {
  let body: unknown = null;
  try {
    const text = await res.text();
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
  } catch {
    body = null;
  }
  return new ApiRequestError(res.status, normalizeApiError(res.status, body, res.statusText));
}

/**
 * Map a validation error to `{fieldName: message}` for inline form display.
 * Understands `detail.errors[] = {loc: [..., field], msg}` (Pydantic), `detail.field`,
 * `detail.fields = {name: msg}`, and the `max_steps_required` code.
 */
export function fieldErrors(err: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!isApiRequestError(err)) return out;
  const detail = err.detail;
  if (detail) {
    const errors = detail.errors;
    if (Array.isArray(errors)) {
      for (const e of errors) {
        if (!isRecord(e)) continue;
        const loc: unknown[] = Array.isArray(e.loc) ? e.loc : [];
        const field = [...loc].reverse().find((p): p is string => typeof p === "string" && p !== "body");
        const msg = typeof e.msg === "string" ? e.msg : err.message;
        if (field && !out[field]) out[field] = msg;
      }
    }
    if (typeof detail.field === "string") out[detail.field] = err.message;
    if (isRecord(detail.fields)) {
      for (const [k, v] of Object.entries(detail.fields)) if (typeof v === "string") out[k] = v;
    }
  }
  if (err.code === "max_steps_required" && !out.max_steps) out.max_steps = err.message;
  return out;
}

/** Directory the SPA is served from, always ending in "/". `/proxy/7860/index.html` -> `/proxy/7860/`. */
export function basePath(pathname: string = window.location.pathname): string {
  if (!pathname || pathname === "/") return "/";
  if (pathname.endsWith("/")) return pathname;
  const last = pathname.slice(pathname.lastIndexOf("/") + 1);
  // A last segment with a dot is a file (index.html); otherwise treat it as a directory.
  return last.includes(".") ? pathname.slice(0, pathname.lastIndexOf("/") + 1) : `${pathname}/`;
}

/** URL for an API route, e.g. `apiUrl("api/env")` -> `/proxy/7860/api/env`. */
export function apiUrl(route: string): string {
  return basePath() + route.replace(/^\/+/, "");
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  json?: unknown;
  body?: BodyInit;
  signal?: AbortSignal;
}

/** Low-level request helper. Returns parsed JSON (or `undefined` for empty bodies). */
export async function request<T>(route: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  let body = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  let res: Response;
  try {
    res = await fetch(apiUrl(route), {
      method: opts.method ?? (body !== undefined ? "POST" : "GET"),
      credentials: "same-origin",
      headers,
      body,
      signal: opts.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiRequestError(0, {
      code: "network_error",
      message: "Cannot reach the AutoTune Studio server.",
      detail: null,
    });
  }
  if (!res.ok) throw await parseErrorResponse(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const enc = encodeURIComponent;

/**
 * `POST /api/datasets/upload` response. The contract does not pin this down yet; the UI accepts
 * either a `DatasetInfo` (has `dataset_id`) or an object with the stored server-side `path`.
 */
export type UploadResponse = Partial<DatasetInfo> & { path?: string; filename?: string };

/** One function per backend route (see fastapi-backend skill). */
export const api = {
  // --- auth / environment
  health: () => request<Record<string, unknown>>("api/health"),
  createSession: (token: string) => request<unknown>("api/auth/session", { method: "POST", json: { token } }),
  env: () => request<EnvironmentInfo>("api/env"),
  hardware: () => request<HardwareInfo>("api/hardware"),
  resources: () => request<ResourceSnapshot>("api/resources"),

  // --- datasets
  createDataset: (spec: DatasetSpec) => request<DatasetInfo>("api/datasets", { method: "POST", json: spec }),
  uploadDataset: (file: File) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    return request<UploadResponse>("api/datasets/upload", { method: "POST", body: fd });
  },
  listDatasets: () => request<DatasetInfo[]>("api/datasets"),
  getDataset: (id: string) => request<DatasetInfo>(`api/datasets/${enc(id)}`),
  previewDataset: (id: string, n = 20) => request<DatasetPreview>(`api/datasets/${enc(id)}/preview?n=${n}`),
  datasetStats: (id: string) => request<StreamStats>(`api/datasets/${enc(id)}/stats`),
  scanDataset: (id: string) => request<unknown>(`api/datasets/${enc(id)}/scan`, { method: "POST" }),
  cancelDataset: (id: string) => request<unknown>(`api/datasets/${enc(id)}/cancel`, { method: "POST" }),

  // --- model
  getModel: () => request<ModelConfig | null>("api/model"),
  putModel: (cfg: ModelConfig) => request<ModelConfig>("api/model", { method: "PUT", json: cfg }),
  suggestedModels: () => request<string[]>("api/models/suggested"),

  // --- training
  getTrainingConfig: () => request<TrainingConfig>("api/training/config"),
  putTrainingConfig: (cfg: TrainingConfig) =>
    request<TrainingConfig>("api/training/config", { method: "PUT", json: cfg }),
  startTraining: (datasetId: string) =>
    request<TrainingStatus>("api/training/start", { method: "POST", json: { dataset_id: datasetId } }),
  stopTraining: () => request<TrainingStatus>("api/training/stop", { method: "POST" }),
  pauseTraining: () => request<TrainingStatus>("api/training/pause", { method: "POST" }),
  resumeTraining: () => request<TrainingStatus>("api/training/resume", { method: "POST" }),
  trainingStatus: () => request<TrainingStatus>("api/training/status"),
  trainingMetrics: () => request<MetricPoint[]>("api/training/metrics"),
  sessions: () => request<SessionSummary[]>("api/sessions"),

  // --- checkpoints / export
  checkpoints: () => request<CheckpointInfo[]>("api/checkpoints"),
  saveCheckpoint: () => request<unknown>("api/checkpoints", { method: "POST" }),
  resumeCheckpoint: (id: string) =>
    request<TrainingStatus>(`api/checkpoints/${enc(id)}/resume`, { method: "POST" }),
  exportModel: (req: ExportRequest) => request<ExportResult>("api/export", { method: "POST", json: req }),
};
