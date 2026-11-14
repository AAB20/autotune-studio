/**
 * Datasets: choose a source, load it (schema discovery from a sample, never a full read),
 * inspect columns / sample preview / streaming stats, run a full streaming scan.
 * The UI never asks for a row count: unknown cardinality is a first-class state.
 */
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, Database, FileUp, Rows3, ScanLine, Table2, Upload } from "lucide-react";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/form";
import { Select } from "../components/ui/select";
import { Badge, EmptyState, Meter, Segmented, Table, Td, Th } from "../components/ui/display";
import { ErrorNotice, PageHeader } from "../components/common";
import { qk, useDatasets } from "../hooks/queries";
import { api, type UploadResponse } from "../services/api";
import { formatBytes, formatCell, formatNumber } from "../services/format";
import { useUiStore } from "../services/uiStore";
import type { Cardinality, DatasetInfo, DatasetSource, DatasetSpec, StreamStats } from "../services/types";

const SOURCES: { value: DatasetSource; label: string }[] = [
  { value: "huggingface", label: "Hugging Face" },
  { value: "csv", label: "CSV" },
  { value: "json", label: "JSON" },
  { value: "jsonl", label: "JSONL" },
  { value: "parquet", label: "Parquet" },
];

const PREVIEW_N = 20;

/** "Unknown — streaming" when rows is null; otherwise the count with exactness. */
export function cardinalityLabel(c: Cardinality | undefined): string {
  if (!c || c.rows === null || c.rows === undefined) return "Unknown — streaming";
  return `${formatNumber(c.rows)} rows${c.exact ? "" : " (approx.)"}`;
}

function sourceFromFilename(name: string): DatasetSource | null {
  const n = name.toLowerCase();
  if (n.endsWith(".csv") || n.endsWith(".csv.gz")) return "csv";
  if (n.endsWith(".jsonl") || n.endsWith(".ndjson") || n.endsWith(".jsonl.gz")) return "jsonl";
  if (n.endsWith(".json") || n.endsWith(".json.gz")) return "json";
  if (n.endsWith(".parquet") || n.endsWith(".pq")) return "parquet";
  return null;
}

const STATE_TONE = { idle: "neutral", streaming: "info", exhausted: "ok", failed: "err", cancelled: "warn" } as const;

function DatasetSourceForm({ onLoaded }: { onLoaded: (info: DatasetInfo) => void }) {
  const [source, setSource] = useState<DatasetSource>("huggingface");
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [split, setSplit] = useState("train");
  const [configName, setConfigName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const isHf = source === "huggingface";

  const load = useMutation({
    mutationFn: (spec: DatasetSpec) => api.createDataset(spec),
    onSuccess: onLoaded,
  });

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadDataset(file),
    onSuccess: (res: UploadResponse, file) => {
      if (res && typeof res.dataset_id === "string" && res.spec && res.columns && res.cardinality) {
        onLoaded(res as DatasetInfo);
        return;
      }
      if (res?.path) {
        const src = sourceFromFilename(file.name) ?? (source === "huggingface" ? "jsonl" : source);
        setSource(src);
        setPath(res.path);
        load.mutate({ source: src, path: res.path, split: split || "train" });
      }
    },
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const spec: DatasetSpec = isHf
      ? { source, name: name.trim(), split: split.trim() || "train", config_name: configName.trim() || null }
      : { source, path: path.trim(), split: split.trim() || "train" };
    load.mutate(spec);
  };

  const canSubmit = isHf ? name.trim().length > 0 : path.trim().length > 0;

  return (
    <Card>
      <CardHeader title="Source" icon={<Database />} />
      <CardBody>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted" id="source-label">
              Source type
            </span>
            <Segmented aria-label="Dataset source type" value={source} onValueChange={setSource} options={SOURCES} />
          </div>
          {isHf ? (
            <Field label="Hugging Face dataset id" hint="e.g. yahma/alpaca-cleaned — streamed, never fully downloaded.">
              {(p) => <Input {...p} value={name} onChange={(e) => setName(e.target.value)} placeholder="org/dataset" spellCheck={false} />}
            </Field>
          ) : (
            <Field label="File or directory path" hint="Must be inside an allowed data root.">
              {(p) => <Input {...p} value={path} onChange={(e) => setPath(e.target.value)} placeholder="/content/data/train.jsonl" spellCheck={false} />}
            </Field>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Split">{(p) => <Input {...p} value={split} onChange={(e) => setSplit(e.target.value)} />}</Field>
            {isHf && (
              <Field label="Config (optional)">
                {(p) => <Input {...p} value={configName} onChange={(e) => setConfigName(e.target.value)} placeholder="default" />}
              </Field>
            )}
          </div>
          <ErrorNotice error={load.error} title="Could not load dataset" />
          <div className="flex flex-wrap gap-2">
            <Button type="submit" variant="primary" loading={load.isPending} disabled={!canSubmit}>
              <ScanLine aria-hidden /> Load &amp; inspect
            </Button>
            <input
              ref={fileRef}
              type="file"
              className="sr-only"
              accept=".csv,.json,.jsonl,.ndjson,.parquet,.pq,.gz"
              aria-label="Upload dataset file"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) upload.mutate(f);
                e.target.value = "";
              }}
            />
            <Button type="button" variant="outline" loading={upload.isPending} onClick={() => fileRef.current?.click()}>
              <Upload aria-hidden /> Upload file
            </Button>
          </div>
          <ErrorNotice error={upload.error} title="Upload failed" />
        </form>
      </CardBody>
    </Card>
  );
}

function StreamStatsCard({ info }: { info: DatasetInfo }) {
  const qc = useQueryClient();
  const id = info.dataset_id;
  const statsQ = useQuery<StreamStats>({
    queryKey: qk.stats(id),
    queryFn: () => api.datasetStats(id),
    refetchInterval: (q) => (q.state.data?.state === "streaming" ? 2000 : false),
  });
  const scan = useMutation({
    mutationFn: () => api.scanDataset(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.stats(id) }),
  });
  const cancel = useMutation({
    mutationFn: () => api.cancelDataset(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.stats(id) }),
  });
  const s = statsQ.data;
  const streaming = s?.state === "streaming";

  return (
    <Card>
      <CardHeader
        title="Streaming stats"
        icon={<Rows3 />}
        actions={
          <>
            <Button size="sm" variant="primary" onClick={() => scan.mutate()} loading={scan.isPending} disabled={streaming}>
              <ScanLine aria-hidden /> Full scan
            </Button>
            <Button size="sm" variant="danger" onClick={() => cancel.mutate()} loading={cancel.isPending} disabled={!streaming}>
              <Ban aria-hidden /> Cancel
            </Button>
          </>
        }
      />
      <CardBody className="flex flex-col gap-3">
        <ErrorNotice error={statsQ.error || scan.error || cancel.error} />
        <div className="flex items-center gap-2">
          <Badge tone={STATE_TONE[s?.state ?? "idle"]}>{s?.state ?? "idle"}</Badge>
          <span className="text-[11px] text-faint">A full scan streams every row once to count and validate; nothing is kept in memory.</span>
        </div>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Rows seen" value={formatNumber(s?.rows_seen ?? 0)} />
          <Stat label="Bytes seen" value={formatBytes(s?.bytes_seen ?? 0)} />
          <Stat label="Batches" value={formatNumber(s?.batches_seen ?? 0)} />
          <Stat label="Malformed" value={<span className={(s?.malformed_rows ?? 0) > 0 ? "text-warn" : undefined}>{formatNumber(s?.malformed_rows ?? 0)}</span>} />
        </dl>
        {s?.progress != null && <Meter label="Scan progress" value={s.progress * 100} tone="accent" />}
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-wider text-faint">Checksum (blake2b, rolling)</p>
          <p className="truncate font-mono text-xs text-muted" title={s?.checksum || undefined}>
            {s?.checksum || "—"}
          </p>
        </div>
        {s?.error && <p className="text-xs text-err">{s.error}</p>}
        {s?.malformed_examples && s.malformed_examples.length > 0 && (
          <details className="text-xs">
            <summary className="cursor-pointer text-warn">Malformed examples ({s.malformed_examples.length})</summary>
            <ul className="mt-1 space-y-1 font-mono text-[11px] text-muted">
              {s.malformed_examples.map((m, i) => (
                <li key={i} className="break-all">
                  {m}
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardBody>
    </Card>
  );
}

function PreviewCard({ id }: { id: string }) {
  const q = useQuery({ queryKey: qk.preview(id), queryFn: () => api.previewDataset(id, PREVIEW_N) });
  const p = q.data;
  return (
    <Card>
      <CardHeader
        title="Sample preview"
        icon={<Table2 />}
        subtitle={p ? `sample · first ${p.rows.length} streamed rows${p.truncated_values ? " · long values truncated" : ""}` : "sample"}
      />
      <ErrorNotice error={q.error} />
      {p && p.rows.length > 0 ? (
        <div className="max-h-[420px] overflow-auto">
          <Table aria-label="Dataset sample preview">
            <caption className="sr-only">Sample of the dataset (not the full data)</caption>
            <thead>
              <tr>
                <Th className="w-10 text-right">#</Th>
                {p.columns.map((c) => (
                  <Th key={c}>{c}</Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {p.rows.map((row, i) => (
                <tr key={i} className="hover:bg-raised/50">
                  <Td className="text-right font-mono text-faint">{i}</Td>
                  {p.columns.map((c) => {
                    const text = formatCell(row[c]);
                    return (
                      <Td key={c} className="max-w-80 truncate font-mono text-xs" title={formatCell(row[c], 2000)}>
                        {text}
                      </Td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      ) : (
        !q.isLoading && <EmptyState title="No sample rows" />
      )}
    </Card>
  );
}

const AUTO = "__auto__";

function DatasetDetails({ info, onReloaded }: { info: DatasetInfo; onReloaded: (i: DatasetInfo) => void }) {
  const override = useMutation({
    mutationFn: (col: string | null) => api.createDataset({ ...info.spec, text_column: col }),
    onSuccess: onReloaded,
  });
  const where = info.spec.source === "huggingface" ? info.spec.name : info.spec.path;
  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Card>
        <CardHeader
          title="Dataset"
          icon={<Database />}
          subtitle={<span className="font-mono">{where}</span>}
          actions={<Badge tone="neutral">{info.spec.source}</Badge>}
        />
        <CardBody className="flex flex-col gap-3">
          <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Detected format" value={<Badge tone={info.detected_format === "unknown" ? "warn" : "accent"}>{info.detected_format ?? "unknown"}</Badge>} mono={false} />
            <Stat
              label="Cardinality"
              mono={false}
              value={<span data-testid="cardinality">{cardinalityLabel(info.cardinality)}</span>}
              hint={info.cardinality.rows == null ? "set max_steps for training" : `via ${info.cardinality.method ?? "unknown"}`}
            />
            <Stat label="Split" value={info.spec.split ?? "train"} />
            <Stat label="Sampled" value={`${formatNumber(info.sampled_rows ?? 0)} rows`} hint={formatBytes(info.sampled_bytes ?? 0)} />
          </dl>
          <Field label="Training text column" hint={`In use: ${(info.text_columns ?? []).join(", ") || "auto-detected"}`} className="max-w-sm">
            {(p) => (
              <Select
                {...p}
                value={info.spec.text_column ?? AUTO}
                onValueChange={(v) => override.mutate(v === AUTO ? null : v)}
                disabled={override.isPending}
                options={[{ value: AUTO, label: "Auto-detect" }, ...info.columns.map((c) => ({ value: c.name, label: c.name, hint: c.dtype }))]}
              />
            )}
          </Field>
          <ErrorNotice error={override.error} title="Override failed" />
          {(info.warnings ?? []).length > 0 && (
            <ul className="space-y-1 rounded-md border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-warn">
              {(info.warnings ?? []).map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Columns" subtitle="statistics from the sample" icon={<Table2 />} />
        <Table aria-label="Dataset columns">
          <thead>
            <tr>
              <Th>Name</Th>
              <Th>Dtype</Th>
              <Th className="text-right">Nulls (sample)</Th>
              <Th>Examples</Th>
            </tr>
          </thead>
          <tbody>
            {info.columns.map((c) => (
              <tr key={c.name} className="hover:bg-raised/50">
                <Td className="font-mono text-xs">
                  {c.name}
                  {(info.text_columns ?? []).includes(c.name) && (
                    <Badge tone="accent" className="ml-1.5">
                      text
                    </Badge>
                  )}
                </Td>
                <Td className="font-mono text-xs text-info">{c.dtype}</Td>
                <Td className="text-right font-mono text-xs">{formatNumber(c.sampled_null_count ?? 0)}</Td>
                <Td className="max-w-96 truncate font-mono text-[11px] text-muted" title={(c.examples ?? []).join("\n")}>
                  {(c.examples ?? []).map((x) => formatCell(x, 60)).join("  ·  ") || "—"}
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      <StreamStatsCard info={info} />
      <PreviewCard id={info.dataset_id} />
    </div>
  );
}

export function DatasetsPage() {
  const qc = useQueryClient();
  const selectedId = useUiStore((s) => s.selectedDatasetId);
  const selectDataset = useUiStore((s) => s.selectDataset);
  const list = useDatasets();
  const current = useQuery({
    queryKey: qk.dataset(selectedId ?? ""),
    queryFn: () => api.getDataset(selectedId as string),
    enabled: !!selectedId,
    retry: false,
  });

  // Forget a stale selection (e.g. backend restarted and no longer knows the id).
  useEffect(() => {
    if (current.error && selectedId) selectDataset(null);
  }, [current.error, selectedId, selectDataset]);

  const onLoaded = (info: DatasetInfo) => {
    qc.setQueryData(qk.dataset(info.dataset_id), info);
    void qc.invalidateQueries({ queryKey: qk.datasets, exact: true });
    void qc.invalidateQueries({ queryKey: qk.preview(info.dataset_id) });
    selectDataset(info.dataset_id);
  };

  const datasets = list.data ?? [];

  return (
    <>
      <PageHeader
        title="Datasets"
        description="Schema and statistics come from a streamed sample; the full dataset is never loaded into memory."
      />
      <div className="grid gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <DatasetSourceForm onLoaded={onLoaded} />
          <Card>
            <CardHeader title="Loaded datasets" icon={<FileUp />} />
            {datasets.length === 0 ? (
              <EmptyState title="Nothing loaded yet" />
            ) : (
              <ul className="max-h-72 overflow-y-auto p-1.5">
                {datasets.map((d) => (
                  <li key={d.dataset_id}>
                    <button
                      type="button"
                      onClick={() => {
                        qc.setQueryData(qk.dataset(d.dataset_id), d);
                        selectDataset(d.dataset_id);
                      }}
                      aria-current={d.dataset_id === selectedId ? "true" : undefined}
                      className="flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-raised aria-[current=true]:bg-raised aria-[current=true]:text-accent"
                    >
                      <span className="truncate font-mono">{d.spec.name ?? d.spec.path}</span>
                      <span className="shrink-0 text-faint">{cardinalityLabel(d.cardinality)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
        <div className="min-w-0">
          {current.data ? (
            <DatasetDetails info={current.data} onReloaded={onLoaded} />
          ) : (
            <Card>
              <EmptyState icon={<Database />} title="No dataset selected">
                Load a Hugging Face dataset, a local file path, or upload a file.
              </EmptyState>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
