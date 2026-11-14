/** Dashboard: environment, hardware/resources, current training status and recent sessions. */
import { Link } from "react-router";
import { Activity, Cpu, HardDrive, History, Server } from "lucide-react";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/card";
import { EmptyState, Meter, Table, Td, Th } from "../components/ui/display";
import { Button } from "../components/ui/button";
import { ErrorNotice, PageHeader, TrainingStateBadge } from "../components/common";
import { ResourceMeters } from "../components/ResourceMeters";
import { useEnv, useHardware, useResources, useSessions, useTrainingStatus } from "../hooks/queries";
import { formatBytes, formatDuration, formatLoss, formatNumber, formatTimestamp, ratioPct } from "../services/format";

export function DashboardPage() {
  const env = useEnv();
  const hw = useHardware();
  const resources = useResources();
  const sessions = useSessions();
  const status = useTrainingStatus();
  const e = env.data;
  const h = hw.data;

  return (
    <>
      <PageHeader title="Dashboard" description="Runtime, hardware and training overview." />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Card>
          <CardHeader title="Environment" icon={<Server />} />
          <CardBody>
            <ErrorNotice error={env.error} />
            {e && (
              <dl className="grid grid-cols-2 gap-3">
                <Stat label="Runtime" value={e.kind} mono={false} hint={e.is_notebook ? "notebook kernel" : "process"} />
                <Stat label="Python" value={e.python_version} />
                <Stat label="Platform" value={e.platform} className="col-span-2" mono={false} />
                <Stat label="Workspace" value={<span title={e.workspace_dir}>{e.workspace_dir}</span>} className="col-span-2" />
                <Stat
                  label="Internet"
                  mono={false}
                  value={e.internet_access == null ? "not probed" : e.internet_access ? "available" : "offline"}
                />
                <Stat label="Data roots" value={formatNumber(e.allowed_roots.length)} hint={e.allowed_roots[0]} />
              </dl>
            )}
            {e?.notes && e.notes.length > 0 && (
              <ul className="mt-3 list-disc space-y-0.5 pl-4 text-[11px] text-faint">
                {e.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="GPU"
            icon={<Cpu />}
            subtitle={h ? (h.cuda_available ? `CUDA ${h.cuda_version ?? "?"} · torch ${h.torch_version ?? "?"}` : "CUDA unavailable") : undefined}
          />
          <CardBody>
            <ErrorNotice error={hw.error} />
            {h && (h.gpus ?? []).length > 0 ? (
              <div className="flex flex-col gap-3">
                {(h.gpus ?? []).map((g) => (
                  <div key={g.index}>
                    <p className="mb-2 truncate text-[13px] font-medium" title={g.name}>
                      <span className="font-mono text-faint">#{g.index}</span> {g.name}
                      {g.compute_capability && <span className="ml-1.5 text-xs text-faint">sm {g.compute_capability}</span>}
                    </p>
                    <ResourceMeters resources={resources} hardware={h} showCpu={false} />
                  </div>
                ))}
                <p className="text-[11px] text-faint">
                  bf16 {h.bf16_supported ? "supported" : "not supported"} · NVML {h.nvml_available ? "on" : "off"}
                </p>
              </div>
            ) : (
              h && <EmptyState icon={<Cpu />} title="No CUDA GPU detected">Training will be slow on CPU.</EmptyState>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="System" icon={<HardDrive />} />
          <CardBody className="flex flex-col gap-3">
            {h && (
              <>
                <Meter
                  label="RAM"
                  value={ratioPct(resources?.ram_used_bytes ?? h.ram_total_bytes - h.ram_available_bytes, resources?.ram_total_bytes ?? h.ram_total_bytes)}
                  detail={`${formatBytes(resources?.ram_used_bytes ?? h.ram_total_bytes - h.ram_available_bytes)} / ${formatBytes(resources?.ram_total_bytes ?? h.ram_total_bytes)}`}
                />
                <Meter label={`CPU · ${h.cpu_count} cores`} value={resources?.cpu_pct ?? null} tone="info" />
                <Meter
                  label="Disk"
                  value={ratioPct(h.disk_total_bytes - h.disk_free_bytes, h.disk_total_bytes)}
                  detail={`${formatBytes(h.disk_free_bytes)} free of ${formatBytes(h.disk_total_bytes)}`}
                />
                <p className="truncate font-mono text-[11px] text-faint" title={h.disk_path}>
                  {h.disk_path}
                </p>
              </>
            )}
          </CardBody>
        </Card>

        <Card className="md:col-span-2 xl:col-span-1">
          <CardHeader
            title="Current training"
            icon={<Activity />}
            actions={
              <Button asChild size="sm" variant="ghost">
                <Link to="/monitor">Open monitor</Link>
              </Button>
            }
          />
          <CardBody>
            <div className="mb-3 flex items-center gap-2">
              <TrainingStateBadge state={status?.state} />
              {status?.backend && <span className="text-xs text-faint">backend: {status.backend}</span>}
            </div>
            <dl className="grid grid-cols-3 gap-3">
              <Stat label="Step" value={`${formatNumber(status?.step ?? 0)}${status?.max_steps ? ` / ${formatNumber(status.max_steps)}` : ""}`} />
              <Stat label="Loss" value={formatLoss(status?.loss)} />
              <Stat label="ETA" value={formatDuration(status?.eta_seconds)} />
            </dl>
            {status?.message && <p className="mt-3 truncate text-xs text-muted">{status.message}</p>}
            {status?.error && <p className="mt-2 text-xs text-err">{status.error}</p>}
          </CardBody>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader title="Recent training sessions" icon={<History />} />
          <ErrorNotice error={sessions.error} />
          {sessions.data && sessions.data.length > 0 ? (
            <div className="max-h-80 overflow-y-auto">
              <Table>
                <thead>
                  <tr>
                    <Th>Job</Th>
                    <Th>Model</Th>
                    <Th>State</Th>
                    <Th>Started</Th>
                    <Th className="text-right">Steps</Th>
                    <Th className="text-right">Final loss</Th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.data.map((s) => (
                    <tr key={s.job_id} className="hover:bg-raised/50">
                      <Td className="font-mono text-xs">{s.job_id.slice(0, 10)}</Td>
                      <Td className="max-w-56 truncate" title={s.model_name}>
                        {s.model_name}
                      </Td>
                      <Td>
                        <TrainingStateBadge state={s.state} />
                      </Td>
                      <Td className="font-mono text-xs text-muted">{formatTimestamp(s.started_at)}</Td>
                      <Td className="text-right font-mono">{formatNumber(s.final_step ?? 0)}</Td>
                      <Td className="text-right font-mono">{formatLoss(s.final_loss)}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          ) : (
            !sessions.isLoading && <EmptyState icon={<History />} title="No training sessions yet" />
          )}
        </Card>
      </div>
    </>
  );
}
