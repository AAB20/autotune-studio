/**
 * Live state fed by the `/ws` event stream (and seeded from REST).
 *
 * `reduceEvent` is a pure function (unit-tested); `useEventStore` wraps it in zustand.
 *
 * Expected `data` payloads per event type (documented for the backend):
 *   status     -> TrainingStatus
 *   metric     -> MetricPoint
 *   resource   -> ResourceSnapshot
 *   log        -> { level?: string, message: string, logger?: string }
 *   dataset    -> { dataset_id: string, stats: StreamStats }  (or StreamStats & { dataset_id })
 *   checkpoint -> CheckpointInfo          (UI only invalidates queries)
 *   export     -> ExportResult            (UI only invalidates queries / shows last result)
 *   error      -> { code?: string, message: string }
 */
import { create } from "zustand";
import type {
  Event as StudioEvent,
  ExportResult,
  MetricPoint,
  ResourceSnapshot,
  StreamStats,
  TrainingStatus,
} from "./types";

export const MAX_LOGS = 1000;
export const MAX_RESOURCE_HISTORY = 300;
export const MAX_METRICS = 5000;

export interface LogLine {
  seq: number;
  ts: number;
  level: string;
  message: string;
  logger?: string;
}

export interface EventState {
  status: TrainingStatus | null;
  metrics: MetricPoint[];
  resources: ResourceSnapshot | null;
  resourceHistory: ResourceSnapshot[];
  logs: LogLine[];
  datasetStats: Record<string, StreamStats>;
  lastError: { code: string; message: string; ts: number } | null;
  lastExport: ExportResult | null;
}

export const initialEventState: EventState = {
  status: null,
  metrics: [],
  resources: null,
  resourceHistory: [],
  logs: [],
  datasetStats: {},
  lastError: null,
  lastExport: null,
};

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Append/merge metric points keeping the series sorted by step and unique per step
 * (a later point for an existing step replaces it, e.g. when eval_loss arrives).
 * When the series exceeds `max`, the older half is decimated (every 2nd point kept)
 * so long runs stay bounded without losing the overall curve shape.
 */
export function mergeMetrics(series: MetricPoint[], incoming: MetricPoint[], max = MAX_METRICS): MetricPoint[] {
  if (incoming.length === 0) return series;
  let out = series.slice();
  for (const p of incoming) {
    if (!p || typeof p.step !== "number") continue;
    const last = out[out.length - 1];
    if (!last || p.step > last.step) {
      out.push(p);
      continue;
    }
    // Out of order or duplicate: binary search for position.
    let lo = 0;
    let hi = out.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((out[mid]?.step ?? 0) < p.step) lo = mid + 1;
      else hi = mid;
    }
    const existing = out[lo];
    if (existing && existing.step === p.step) out[lo] = { ...existing, ...stripNulls(p) };
    else out.splice(lo, 0, p);
  }
  while (out.length > max) {
    const half = Math.floor(out.length / 2);
    const older = out.slice(0, half).filter((_, i) => i % 2 === 0);
    out = older.concat(out.slice(half));
  }
  return out;
}

function stripNulls(p: MetricPoint): Partial<MetricPoint> {
  const o: Partial<MetricPoint> = {};
  for (const [k, v] of Object.entries(p)) if (v !== null && v !== undefined) (o as Record<string, unknown>)[k] = v;
  return o;
}

function pushBounded<T>(list: T[], item: T, max: number): T[] {
  const out = list.length >= max ? list.slice(list.length - max + 1) : list.slice();
  out.push(item);
  return out;
}

/** Pure reducer: apply one server event to the state. Unknown types are ignored. */
export function reduceEvent(state: EventState, ev: StudioEvent): EventState {
  const data: Record<string, unknown> = isRecord(ev.data) ? ev.data : {};
  const ts = typeof ev.ts === "number" ? ev.ts : Date.now() / 1000;
  switch (ev.type) {
    case "status": {
      const status = data as TrainingStatus;
      const prevJob = state.status?.job_id ?? null;
      const newJob = status.job_id ?? null;
      // A different job started: its metric series starts fresh.
      const metrics = newJob && prevJob && newJob !== prevJob ? [] : state.metrics;
      return { ...state, status, metrics };
    }
    case "metric":
      return { ...state, metrics: mergeMetrics(state.metrics, [data as unknown as MetricPoint]) };
    case "resource": {
      const snap = data as unknown as ResourceSnapshot;
      return { ...state, resources: snap, resourceHistory: pushBounded(state.resourceHistory, snap, MAX_RESOURCE_HISTORY) };
    }
    case "log": {
      const line: LogLine = {
        seq: ev.seq,
        ts,
        level: typeof data.level === "string" ? data.level.toLowerCase() : "info",
        message: typeof data.message === "string" ? data.message : JSON.stringify(data),
        logger: typeof data.logger === "string" ? data.logger : undefined,
      };
      return { ...state, logs: pushBounded(state.logs, line, MAX_LOGS) };
    }
    case "dataset": {
      const id = typeof data.dataset_id === "string" ? data.dataset_id : null;
      if (!id) return state;
      const stats = (isRecord(data.stats) ? data.stats : data) as StreamStats;
      return { ...state, datasetStats: { ...state.datasetStats, [id]: stats } };
    }
    case "export":
      return typeof data.path === "string" ? { ...state, lastExport: data as unknown as ExportResult } : state;
    case "error": {
      const message = typeof data.message === "string" ? data.message : "Unknown server error";
      const code = typeof data.code === "string" ? data.code : "error";
      const line: LogLine = { seq: ev.seq, ts, level: "error", message };
      return { ...state, lastError: { code, message, ts }, logs: pushBounded(state.logs, line, MAX_LOGS) };
    }
    default:
      return state;
  }
}

export interface EventStore extends EventState {
  /** Client time (ms) of the last status update, used so stale REST responses never clobber WS data. */
  statusUpdatedAt: number;
  apply: (ev: StudioEvent) => void;
  /** Set status from REST; ignored if a newer status (by `at`, ms) is already stored. */
  setStatus: (s: TrainingStatus, at?: number) => void;
  seedMetrics: (points: MetricPoint[]) => void;
  setResources: (r: ResourceSnapshot) => void;
  setDatasetStats: (id: string, s: StreamStats) => void;
  clearLogs: () => void;
  reset: () => void;
}

export const useEventStore = create<EventStore>()((set) => ({
  ...initialEventState,
  statusUpdatedAt: 0,
  apply: (ev) =>
    set((s) => {
      const next = reduceEvent(s, ev);
      return ev.type === "status" ? { ...next, statusUpdatedAt: Date.now() } : next;
    }),
  setStatus: (status, at = Date.now()) =>
    set((s) => (at < s.statusUpdatedAt ? s : { status, statusUpdatedAt: at })),
  seedMetrics: (points) => set((s) => ({ metrics: mergeMetrics(s.metrics, points) })),
  setResources: (r) =>
    set((s) =>
      s.resources && (s.resources.timestamp ?? 0) >= (r.timestamp ?? 0)
        ? s
        : { resources: r, resourceHistory: pushBounded(s.resourceHistory, r, MAX_RESOURCE_HISTORY) },
    ),
  setDatasetStats: (id, stats) => set((s) => ({ datasetStats: { ...s.datasetStats, [id]: stats } })),
  clearLogs: () => set({ logs: [] }),
  reset: () => set({ ...initialEventState, statusUpdatedAt: 0 }),
}));
