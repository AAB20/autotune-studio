import { describe, expect, it } from "vitest";
import {
  MAX_LOGS,
  MAX_RESOURCE_HISTORY,
  initialEventState,
  mergeMetrics,
  reduceEvent,
  type EventState,
} from "../services/eventStore";
import type { Event as StudioEvent, MetricPoint } from "../services/types";

let seq = 0;
const ev = (type: StudioEvent["type"], data: Record<string, unknown>): StudioEvent => ({ seq: ++seq, type, ts: 1000 + seq, data });
const apply = (s: EventState, ...evs: StudioEvent[]) => evs.reduce(reduceEvent, s);

describe("reduceEvent", () => {
  it("appends metric points in step order and merges duplicates", () => {
    let s = apply(
      initialEventState,
      ev("metric", { step: 1, loss: 2.0 }),
      ev("metric", { step: 2, loss: 1.8 }),
      ev("metric", { step: 3, loss: 1.6 }),
    );
    expect(s.metrics.map((m) => m.step)).toEqual([1, 2, 3]);
    // eval_loss for step 2 arrives later and is merged, not duplicated.
    s = reduceEvent(s, ev("metric", { step: 2, eval_loss: 1.9, loss: null }));
    expect(s.metrics).toHaveLength(3);
    expect(s.metrics[1]).toMatchObject({ step: 2, loss: 1.8, eval_loss: 1.9 });
    // Out-of-order point is inserted in place.
    s = reduceEvent(s, ev("metric", { step: 0, loss: 2.5 }));
    expect(s.metrics.map((m) => m.step)).toEqual([0, 1, 2, 3]);
  });

  it("does not mutate the previous state", () => {
    const before = apply(initialEventState, ev("metric", { step: 1, loss: 1 }));
    const after = reduceEvent(before, ev("metric", { step: 2, loss: 1 }));
    expect(before.metrics).toHaveLength(1);
    expect(after.metrics).toHaveLength(2);
  });

  it("bounds the log list to MAX_LOGS keeping the newest lines", () => {
    let s = initialEventState;
    for (let i = 0; i < MAX_LOGS + 250; i++) s = reduceEvent(s, ev("log", { level: "INFO", message: `line ${i}` }));
    expect(s.logs).toHaveLength(MAX_LOGS);
    expect(s.logs[0]?.message).toBe("line 250");
    expect(s.logs.at(-1)?.message).toBe(`line ${MAX_LOGS + 249}`);
    expect(s.logs.at(-1)?.level).toBe("info");
  });

  it("keeps latest resources and a bounded history", () => {
    let s = initialEventState;
    for (let i = 0; i < MAX_RESOURCE_HISTORY + 10; i++) {
      s = reduceEvent(s, ev("resource", { cpu_pct: i, ram_used_bytes: 1, ram_total_bytes: 2, timestamp: i }));
    }
    expect(s.resourceHistory).toHaveLength(MAX_RESOURCE_HISTORY);
    expect(s.resources?.cpu_pct).toBe(MAX_RESOURCE_HISTORY + 9);
  });

  it("stores status and resets metrics when a new job starts", () => {
    let s = apply(initialEventState, ev("status", { state: "running", job_id: "a", step: 1 }), ev("metric", { step: 1, loss: 1 }));
    expect(s.status?.state).toBe("running");
    s = reduceEvent(s, ev("status", { state: "running", job_id: "a", step: 2 }));
    expect(s.metrics).toHaveLength(1);
    s = reduceEvent(s, ev("status", { state: "preparing", job_id: "b" }));
    expect(s.metrics).toHaveLength(0);
  });

  it("stores dataset stats by id (nested or flat payload)", () => {
    let s = reduceEvent(initialEventState, ev("dataset", { dataset_id: "d1", stats: { state: "streaming", rows_seen: 10 } }));
    s = reduceEvent(s, ev("dataset", { dataset_id: "d2", state: "exhausted", rows_seen: 5 }));
    expect(s.datasetStats.d1?.rows_seen).toBe(10);
    expect(s.datasetStats.d2?.state).toBe("exhausted");
    expect(reduceEvent(s, ev("dataset", { stats: {} }))).toBe(s);
  });

  it("records errors as lastError and an error log line; ignores unknown types", () => {
    const s = reduceEvent(initialEventState, ev("error", { code: "out_of_memory", message: "CUDA OOM" }));
    expect(s.lastError).toMatchObject({ code: "out_of_memory", message: "CUDA OOM" });
    expect(s.logs.at(-1)).toMatchObject({ level: "error", message: "CUDA OOM" });
    expect(reduceEvent(s, ev("hello", {}))).toBe(s);
    expect(reduceEvent(s, ev("checkpoint", { checkpoint_id: "c" }))).toBe(s);
  });
});

describe("mergeMetrics decimation", () => {
  it("keeps the series bounded while preserving the first and latest points", () => {
    const pts: MetricPoint[] = Array.from({ length: 300 }, (_, i) => ({ step: i + 1, loss: 1 / (i + 1) }));
    const out = mergeMetrics([], pts, 100);
    expect(out.length).toBeLessThanOrEqual(100);
    expect(out[0]?.step).toBe(1);
    expect(out.at(-1)?.step).toBe(300);
    const steps = out.map((p) => p.step);
    expect([...steps].sort((a, b) => a - b)).toEqual(steps);
  });
});
