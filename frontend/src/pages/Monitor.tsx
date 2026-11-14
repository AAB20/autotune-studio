/** Live training monitor: loss / LR charts, progress, resources, logs and job controls. */
import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Pause, Play, Square, Terminal, Trash2 } from "lucide-react";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { EmptyState, Meter } from "../components/ui/display";
import { ErrorNotice, PageHeader, TrainingStateBadge } from "../components/common";
import { LogConsole } from "../components/LogConsole";
import { ResourceMeters } from "../components/ResourceMeters";
import { qk, useHardware, useMetrics, useResources, useTrainingStatus } from "../hooks/queries";
import { api } from "../services/api";
import { useEventStore } from "../services/eventStore";
import { formatDuration, formatLoss, formatNumber, formatPct, formatSci } from "../services/format";
import type { MetricPoint, TrainingStatus } from "../services/types";

const AXIS = { stroke: "var(--color-faint)", fontSize: 11, tickLine: false } as const;

function Chart({
  data,
  lines,
  yFormat,
  label,
}: {
  data: MetricPoint[];
  lines: { key: keyof MetricPoint; name: string; color: string }[];
  yFormat: (v: number) => string;
  label: string;
}) {
  const has = data.some((p) => lines.some((l) => typeof p[l.key] === "number"));
  if (!has) return <EmptyState icon={<Activity />} title={`No ${label} data yet`} />;
  return (
    <div className="h-56 w-full" role="img" aria-label={`${label} chart over training steps`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--color-edge)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...AXIS} tickFormatter={(v: number) => formatNumber(v)} />
          <YAxis {...AXIS} width={52} tickFormatter={yFormat} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "var(--color-raised)", border: "1px solid var(--color-edge-strong)", borderRadius: 6, fontSize: 12 }}
            labelStyle={{ color: "var(--color-muted)" }}
            labelFormatter={(v) => `step ${formatNumber(Number(v))}`}
            formatter={(v) => (typeof v === "number" ? yFormat(v) : String(v))}
          />
          {lines.map((l) => (
            <Line
              key={String(l.key)}
              type="monotone"
              dataKey={l.key as string}
              name={l.name}
              stroke={l.color}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Seconds since `started_at`, ticking once per second while running. */
function useElapsed(status: TrainingStatus | null): number | null {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const running = status?.state === "running" || status?.state === "preparing";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [running]);
  return status?.started_at ? Math.max(0, now - status.started_at) : null;
}

export function MonitorPage() {
  const qc = useQueryClient();
  const status = useTrainingStatus();
  const metrics = useMetrics();
  const resources = useResources();
  const hw = useHardware();
  const clearLogs = useEventStore((s) => s.clearLogs);
  const elapsed = useElapsed(status);

  const onDone = (st: TrainingStatus | undefined) => {
    if (st && typeof st === "object" && "state" in st) useEventStore.getState().setStatus(st);
    void qc.invalidateQueries({ queryKey: qk.trainingStatus });
  };
  const pause = useMutation({ mutationFn: api.pauseTraining, onSuccess: onDone });
  const resume = useMutation({ mutationFn: api.resumeTraining, onSuccess: onDone });
  const stop = useMutation({ mutationFn: api.stopTraining, onSuccess: onDone });

  const state = status?.state ?? "idle";
  const supportsPause = status?.supports_pause ?? true;
  const canPause = state === "running" && supportsPause;
  const canResume = state === "paused";
  const canStop = state === "running" || state === "paused" || state === "preparing";
  const progress = status?.max_steps ? ((status.step ?? 0) / status.max_steps) * 100 : null;
  const gpuUtil = resources?.gpu_util_pct ?? null;

  return (
    <>
      <PageHeader
        title="Monitor"
        description={status?.job_id ? <span className="font-mono">job {status.job_id}</span> : "No active job"}
        actions={
          <>
            <TrainingStateBadge state={state} />
            <Button
              size="sm"
              onClick={() => pause.mutate()}
              loading={pause.isPending}
              disabled={!canPause}
              title={!supportsPause ? "This training backend does not support pausing" : undefined}
            >
              <Pause aria-hidden /> Pause
            </Button>
            <Button size="sm" onClick={() => resume.mutate()} loading={resume.isPending} disabled={!canResume}>
              <Play aria-hidden /> Resume
            </Button>
            <Button size="sm" variant="danger" onClick={() => stop.mutate()} loading={stop.isPending} disabled={!canStop}>
              <Square aria-hidden /> Stop
            </Button>
          </>
        }
      />
      {!supportsPause && state === "running" && (
        <p className="-mt-2 mb-3 text-[11px] text-faint">Pause is not supported by the current backend ({status?.backend ?? "unknown"}).</p>
      )}
      <ErrorNotice error={pause.error || resume.error || stop.error} title="Control request failed" />

      <Card className="mb-4">
        <CardBody>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            <Stat
              label="Step"
              value={
                <span data-testid="step">
                  {formatNumber(status?.step ?? 0)}
                  <span className="text-faint"> / {status?.max_steps ? formatNumber(status.max_steps) : "—"}</span>
                </span>
              }
            />
            <Stat label="Epoch" value={status?.epoch == null ? "—" : status.epoch.toFixed(2)} />
            <Stat label="Loss" value={formatLoss(status?.loss)} />
            <Stat label="Learning rate" value={formatSci(status?.learning_rate)} />
            <Stat label="ETA" value={<span data-testid="eta">{formatDuration(status?.eta_seconds)}</span>} hint={elapsed != null ? `elapsed ${formatDuration(elapsed)}` : undefined} />
            <Stat
              label="GPU util"
              value={formatPct(gpuUtil)}
              hint={status?.effective_batch_size ? `batch ${status.micro_batch_size ?? "?"}×acc = ${status.effective_batch_size}` : undefined}
            />
          </dl>
          {progress !== null && <Meter className="mt-4" label="Progress" value={progress} tone="accent" />}
          {status?.message && <p className="mt-3 truncate text-xs text-muted">{status.message}</p>}
          {status?.error && (
            <p role="alert" className="mt-2 text-xs text-err">
              {status.error}
            </p>
          )}
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-[1fr_1fr_320px]">
        <Card>
          <CardHeader title="Loss" subtitle="training (amber) · eval (blue)" />
          <CardBody className="px-1 pb-2">
            <Chart
              data={metrics}
              label="loss"
              yFormat={(v) => v.toFixed(3)}
              lines={[
                { key: "loss", name: "loss", color: "var(--color-accent)" },
                { key: "eval_loss", name: "eval loss", color: "var(--color-info)" },
              ]}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Learning rate" />
          <CardBody className="px-1 pb-2">
            <Chart
              data={metrics}
              label="learning rate"
              yFormat={(v) => formatSci(v, 1)}
              lines={[{ key: "learning_rate", name: "lr", color: "var(--color-ok)" }]}
            />
          </CardBody>
        </Card>
        <Card className="lg:col-span-2 2xl:col-span-1">
          <CardHeader title="Resources" />
          <CardBody>
            <ResourceMeters resources={resources} hardware={hw.data} />
          </CardBody>
        </Card>
      </div>

      <Card className="mt-4 overflow-hidden">
        <CardHeader
          title="Log"
          icon={<Terminal />}
          actions={
            <Button size="sm" variant="ghost" onClick={clearLogs} aria-label="Clear log console">
              <Trash2 aria-hidden /> Clear
            </Button>
          }
        />
        <LogConsole />
      </Card>
    </>
  );
}
