/** Checkpoint history, save-now, resume, and export (LoRA adapter / merged 16-bit). */
import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { Download, HardDriveDownload, Play, Save } from "lucide-react";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/form";
import { Select } from "../components/ui/select";
import { Badge, EmptyState, Table, Td, Th } from "../components/ui/display";
import { ErrorNotice, PageHeader } from "../components/common";
import { qk, useCheckpoints, useTrainingStatus } from "../hooks/queries";
import { api } from "../services/api";
import { formatBytes, formatNumber, formatTimestamp } from "../services/format";
import { useUiStore } from "../services/uiStore";
import type { ExportResult } from "../services/types";

type ExportFormat = ExportResult["format"];

const LATEST = "__latest__";

function ExportPanel() {
  const checkpoints = useCheckpoints();
  const [ckpt, setCkpt] = useState(LATEST);
  const [format, setFormat] = useState<ExportFormat>("lora_adapter");
  const [outputDir, setOutputDir] = useState("");
  const [result, setResult] = useState<ExportResult | null>(null);
  const exp = useMutation({
    mutationFn: () =>
      api.exportModel({ checkpoint_id: ckpt === LATEST ? null : ckpt, format, output_dir: outputDir.trim() || null }),
    onSuccess: setResult,
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    setResult(null);
    exp.mutate();
  };
  return (
    <Card>
      <CardHeader title="Export" icon={<Download />} />
      <CardBody>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <Field label="Checkpoint">
            {(p) => (
              <Select
                {...p}
                value={ckpt}
                onValueChange={setCkpt}
                options={[
                  { value: LATEST, label: "Latest checkpoint" },
                  ...(checkpoints.data ?? []).map((c) => ({ value: c.checkpoint_id, label: `step ${c.step}`, hint: c.checkpoint_id.slice(0, 8) })),
                ]}
              />
            )}
          </Field>
          <Field
            label="Format"
            hint={format === "lora_adapter" ? "Adapter weights only (small, needs the base model)." : "Adapter merged into base weights, 16-bit (large)."}
          >
            {(p) => (
              <Select
                {...p}
                value={format}
                onValueChange={(v) => setFormat(v as ExportFormat)}
                options={[
                  { value: "lora_adapter", label: "LoRA adapter" },
                  { value: "merged_16bit", label: "Merged 16-bit" },
                ]}
              />
            )}
          </Field>
          <Field label="Output directory (optional)" hint="Defaults to the workspace export folder.">
            {(p) => <Input {...p} value={outputDir} onChange={(e) => setOutputDir(e.target.value)} spellCheck={false} />}
          </Field>
          <ErrorNotice error={exp.error} title="Export failed" />
          <div>
            <Button type="submit" variant="primary" loading={exp.isPending}>
              <Download aria-hidden /> Export
            </Button>
          </div>
        </form>
        {result && (
          <div className="mt-4 rounded-md border border-ok/30 bg-ok/5 p-3" aria-live="polite">
            <dl className="grid grid-cols-2 gap-3">
              <Stat label="Format" value={result.format} />
              <Stat label="Size" value={formatBytes(result.size_bytes)} />
              <Stat label="Path" value={<span title={result.path}>{result.path}</span>} className="col-span-2" />
            </dl>
            <p className="mt-3 text-[11px] uppercase tracking-wider text-faint">Files ({result.files.length})</p>
            <ul className="mt-1 max-h-40 overflow-y-auto font-mono text-[11px] text-muted">
              {result.files.map((f) => (
                <li key={f} className="truncate" title={f}>
                  {f}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

export function CheckpointsPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const toast = useUiStore((s) => s.toast);
  const checkpoints = useCheckpoints();
  const status = useTrainingStatus();
  const running = status?.state === "running" || status?.state === "paused";

  const save = useMutation({
    mutationFn: api.saveCheckpoint,
    onSuccess: () => {
      toast({ kind: "info", title: "Checkpoint requested", message: "It will be written at the next safe step." });
      void qc.invalidateQueries({ queryKey: qk.checkpoints });
    },
  });
  const resume = useMutation({
    mutationFn: (id: string) => api.resumeCheckpoint(id),
    onSuccess: () => {
      toast({ kind: "success", title: "Resuming from checkpoint" });
      void qc.invalidateQueries({ queryKey: qk.trainingStatus });
      navigate("/monitor");
    },
  });

  const list = [...(checkpoints.data ?? [])].sort((a, b) => b.created_at - a.created_at);

  return (
    <>
      <PageHeader
        title="Checkpoints"
        description="Adapter checkpoints with the data-stream cursor, so training resumes exactly where it stopped."
        actions={
          <Button onClick={() => save.mutate()} loading={save.isPending} disabled={!running} title={running ? undefined : "No training job is running"}>
            <Save aria-hidden /> Save now
          </Button>
        }
      />
      <ErrorNotice error={save.error || resume.error} title="Checkpoint action failed" />
      <div className="mt-2 grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card>
          <CardHeader title="History" icon={<HardDriveDownload />} />
          <ErrorNotice error={checkpoints.error} />
          {list.length > 0 ? (
            <Table aria-label="Checkpoints">
              <thead>
                <tr>
                  <Th className="text-right">Step</Th>
                  <Th>Created</Th>
                  <Th className="text-right">Size</Th>
                  <Th>Path</Th>
                  <Th>
                    <span className="sr-only">Actions</span>
                  </Th>
                </tr>
              </thead>
              <tbody>
                {list.map((c) => (
                  <tr key={c.checkpoint_id} className="hover:bg-raised/50">
                    <Td className="text-right font-mono">
                      {formatNumber(c.step)}
                      {c.is_final && (
                        <Badge tone="accent" className="ml-1.5">
                          final
                        </Badge>
                      )}
                    </Td>
                    <Td className="font-mono text-xs text-muted">{formatTimestamp(c.created_at)}</Td>
                    <Td className="text-right font-mono text-xs">{formatBytes(c.size_bytes)}</Td>
                    <Td className="max-w-72 truncate font-mono text-[11px] text-muted" title={c.path}>
                      {c.path}
                    </Td>
                    <Td className="text-right">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => resume.mutate(c.checkpoint_id)}
                        loading={resume.isPending && resume.variables === c.checkpoint_id}
                        disabled={running}
                        aria-label={`Resume training from step ${c.step}`}
                      >
                        <Play aria-hidden /> Resume
                      </Button>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            !checkpoints.isLoading && <EmptyState icon={<HardDriveDownload />} title="No checkpoints yet" />
          )}
        </Card>
        <ExportPanel />
      </div>
    </>
  );
}
