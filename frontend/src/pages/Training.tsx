/** Training configuration (LoRA / QLoRA + hyper-parameters) and job start. */
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { Play, Save, SlidersHorizontal } from "lucide-react";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/form";
import { Select } from "../components/ui/select";
import { Badge, Segmented } from "../components/ui/display";
import { ErrorNotice, PageHeader } from "../components/common";
import { qk, useDatasets, useModelConfig, useTrainingConfig, useTrainingStatus } from "../hooks/queries";
import { api, fieldErrors } from "../services/api";
import { useUiStore } from "../services/uiStore";
import { cardinalityLabel } from "./Datasets";
import type { DatasetInfo, TrainingConfig } from "../services/types";

type Method = NonNullable<TrainingConfig["method"]>;
type Opt = NonNullable<TrainingConfig["optimizer"]>;
type Sched = NonNullable<TrainingConfig["lr_scheduler"]>;

const OPTIMIZERS: Opt[] = ["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "sgd", "adafactor"];
const SCHEDULERS: Sched[] = ["linear", "cosine", "constant", "constant_with_warmup"];

/** Numeric fields edited as strings. `nullable` fields accept an empty value (sent as null). */
const NUMERIC = {
  learning_rate: { label: "Learning rate", int: false, nullable: false, step: "any" },
  num_epochs: { label: "Epochs", int: false, nullable: true, step: "any" },
  max_steps: { label: "Max steps", int: true, nullable: true, step: "1" },
  per_device_batch_size: { label: "Batch size (per device)", int: true, nullable: false, step: "1" },
  gradient_accumulation_steps: { label: "Gradient accumulation", int: true, nullable: false, step: "1" },
  lora_r: { label: "LoRA r", int: true, nullable: false, step: "1" },
  lora_alpha: { label: "LoRA alpha", int: true, nullable: false, step: "1" },
  lora_dropout: { label: "LoRA dropout", int: false, nullable: false, step: "any" },
  warmup_steps: { label: "Warmup steps", int: true, nullable: false, step: "1" },
  save_steps: { label: "Save every N steps", int: true, nullable: false, step: "1" },
  val_fraction: { label: "Validation fraction", int: false, nullable: false, step: "any" },
} as const;

type NumKey = keyof typeof NUMERIC;
type FormNums = Record<NumKey, string>;

const DEFAULTS: TrainingConfig = {
  method: "qlora",
  learning_rate: 2e-4,
  num_epochs: 1,
  max_steps: null,
  per_device_batch_size: 2,
  gradient_accumulation_steps: 4,
  lora_r: 16,
  lora_alpha: 16,
  lora_dropout: 0,
  optimizer: "adamw_torch",
  lr_scheduler: "linear",
  warmup_steps: 5,
  save_steps: 50,
  val_fraction: 0,
};

function toForm(cfg: TrainingConfig): FormNums {
  const out = {} as FormNums;
  for (const k of Object.keys(NUMERIC) as NumKey[]) {
    const v = cfg[k] ?? DEFAULTS[k];
    out[k] = v === null || v === undefined ? "" : String(v);
  }
  return out;
}

/** Convert form strings to numbers; returns client-side errors for unparsable values. */
export function parseNumbers(form: FormNums): { values: Partial<TrainingConfig>; errors: Record<string, string> } {
  const values: Record<string, number | null> = {};
  const errors: Record<string, string> = {};
  for (const k of Object.keys(NUMERIC) as NumKey[]) {
    const meta = NUMERIC[k];
    const raw = form[k].trim();
    if (raw === "") {
      if (meta.nullable) values[k] = null;
      else errors[k] = "Required";
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) errors[k] = "Not a number";
    else if (meta.int && !Number.isInteger(n)) errors[k] = "Must be an integer";
    else values[k] = n;
  }
  return { values: values as Partial<TrainingConfig>, errors };
}

export function TrainingPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const toast = useUiStore((s) => s.toast);
  const selectedId = useUiStore((s) => s.selectedDatasetId);
  const selectDataset = useUiStore((s) => s.selectDataset);
  const cfgQ = useTrainingConfig();
  const datasets = useDatasets();
  const model = useModelConfig();
  const status = useTrainingStatus();

  const [base, setBase] = useState<TrainingConfig>(DEFAULTS);
  const [method, setMethod] = useState<Method>("qlora");
  const [optimizer, setOptimizer] = useState<Opt>("adamw_torch");
  const [scheduler, setScheduler] = useState<Sched>("linear");
  const [nums, setNums] = useState<FormNums>(() => toForm(DEFAULTS));
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    const c = cfgQ.data;
    if (!c) return;
    setBase(c);
    setMethod(c.method ?? "qlora");
    setOptimizer(c.optimizer ?? "adamw_torch");
    setScheduler(c.lr_scheduler ?? "linear");
    setNums(toForm(c));
  }, [cfgQ.data]);

  const selected: DatasetInfo | undefined = useMemo(
    () => (datasets.data ?? []).find((d) => d.dataset_id === selectedId),
    [datasets.data, selectedId],
  );
  // Fall back to the dataset cached by the Datasets page when the list is not loaded yet.
  const cachedSelected = useQuery<DatasetInfo>({
    queryKey: qk.dataset(selectedId ?? ""),
    queryFn: () => api.getDataset(selectedId as string),
    enabled: !!selectedId && !selected && !datasets.isLoading,
    retry: false,
  });
  const dataset = selected ?? cachedSelected.data;
  const unknownSize = !!dataset && (dataset.cardinality.rows === null || dataset.cardinality.rows === undefined);

  const buildConfig = (): TrainingConfig | null => {
    const { values, errors } = parseNumbers(nums);
    if (unknownSize && values.max_steps == null && !errors.max_steps) {
      errors.max_steps = "Required: dataset size unknown";
    }
    if (values.max_steps == null && values.num_epochs == null && !errors.max_steps) {
      errors.num_epochs = "Set epochs or max steps";
    }
    setClientErrors(errors);
    if (Object.keys(errors).length > 0) return null;
    return { ...base, ...values, method, optimizer, lr_scheduler: scheduler };
  };

  const saveCfg = useMutation({
    mutationFn: (cfg: TrainingConfig) => api.putTrainingConfig(cfg),
    onSuccess: (cfg) => {
      if (cfg) qc.setQueryData(qk.trainingConfig, cfg);
    },
  });
  const start = useMutation({
    mutationFn: async ({ cfg, datasetId }: { cfg: TrainingConfig; datasetId: string }) => {
      const saved = await api.putTrainingConfig(cfg);
      if (saved) qc.setQueryData(qk.trainingConfig, saved);
      return api.startTraining(datasetId);
    },
    onSuccess: (st) => {
      if (st) qc.setQueryData(qk.trainingStatus, st);
      void qc.invalidateQueries({ queryKey: qk.trainingStatus });
      toast({ kind: "success", title: "Training started" });
      navigate("/monitor");
    },
  });

  const serverErrors = { ...fieldErrors(saveCfg.error), ...fieldErrors(start.error) };
  const errorFor = (k: string) => clientErrors[k] ?? serverErrors[k] ?? null;
  const generalError = [saveCfg.error, start.error].find((e) => e && Object.keys(fieldErrors(e)).length === 0) ?? null;
  const active = status && ["preparing", "running", "paused", "stopping"].includes(status.state ?? "idle");

  const onSave = (e: FormEvent) => {
    e.preventDefault();
    const cfg = buildConfig();
    if (cfg) {
      start.reset();
      saveCfg.mutate(cfg, { onSuccess: () => toast({ kind: "success", title: "Training configuration saved" }) });
    }
  };
  const onStart = () => {
    const cfg = buildConfig();
    if (!cfg || !selectedId) return;
    saveCfg.reset();
    start.mutate({ cfg, datasetId: selectedId });
  };

  const num = (k: NumKey, hint?: string) => {
    const meta = NUMERIC[k];
    return (
      <Field label={meta.label} error={errorFor(k)} hint={hint}>
        {(p) => (
          <Input
            {...p}
            name={k}
            type="number"
            step={meta.step}
            inputMode={meta.int ? "numeric" : "decimal"}
            value={nums[k]}
            placeholder={meta.nullable ? "unset" : undefined}
            onChange={(e) => setNums((s) => ({ ...s, [k]: e.target.value }))}
          />
        )}
      </Field>
    );
  };

  const effBatch = Number(nums.per_device_batch_size) * Number(nums.gradient_accumulation_steps);

  return (
    <>
      <PageHeader
        title="Training"
        description="Parameter-efficient fine-tuning. The configuration is validated by the server before a job starts."
      />
      <form onSubmit={onSave} className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="flex min-w-0 flex-col gap-4">
          <Card>
            <CardHeader title="Method & schedule" icon={<SlidersHorizontal />} />
            <CardBody className="flex flex-col gap-3.5">
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-muted">Method</span>
                <Segmented
                  aria-label="Training method"
                  value={method}
                  onValueChange={setMethod}
                  options={[
                    { value: "qlora", label: "QLoRA (4-bit base)" },
                    { value: "lora", label: "LoRA (16-bit base)" },
                  ]}
                />
                {method === "qlora" && model.data && model.data.load_in_4bit === false && (
                  <p className="text-[11px] text-warn">The model is configured without 4-bit loading; QLoRA expects load_in_4bit.</p>
                )}
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {num("learning_rate")}
                {num("num_epochs", "Used when dataset length is known")}
                {num("max_steps", unknownSize ? "required: dataset size unknown" : "Optional; overrides epochs")}
                {num("warmup_steps")}
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Optimizer" error={errorFor("optimizer")}>
                  {(p) => <Select {...p} value={optimizer} onValueChange={(v) => setOptimizer(v as Opt)} options={OPTIMIZERS.map((o) => ({ value: o, label: o }))} />}
                </Field>
                <Field label="LR scheduler" error={errorFor("lr_scheduler")}>
                  {(p) => <Select {...p} value={scheduler} onValueChange={(v) => setScheduler(v as Sched)} options={SCHEDULERS.map((o) => ({ value: o, label: o }))} />}
                </Field>
              </div>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Batching & adapters" />
            <CardBody className="flex flex-col gap-3.5">
              <div className="grid gap-3 sm:grid-cols-3">
                {num("per_device_batch_size")}
                {num("gradient_accumulation_steps")}
                <div className="flex flex-col justify-end pb-1 text-xs text-muted">
                  Effective batch: <span className="font-mono text-[15px] text-fg">{Number.isFinite(effBatch) ? effBatch : "—"}</span>
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                {num("lora_r")}
                {num("lora_alpha")}
                {num("lora_dropout")}
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                {num("save_steps")}
                {num("val_fraction", "0 – 0.5, deterministic hash split")}
              </div>
            </CardBody>
          </Card>
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader title="Run" icon={<Play />} />
            <CardBody className="flex flex-col gap-3">
              <Field label="Dataset" hint={dataset ? cardinalityLabel(dataset.cardinality) : "Load one on the Datasets page"}>
                {(p) => (
                  <Select
                    {...p}
                    value={selectedId ?? ""}
                    onValueChange={(v) => selectDataset(v)}
                    placeholder="Select a dataset"
                    options={(datasets.data ?? []).map((d) => ({ value: d.dataset_id, label: d.spec.name ?? d.spec.path ?? d.dataset_id }))}
                  />
                )}
              </Field>
              {unknownSize && (
                <Badge tone="warn" className="self-start">
                  Unknown size: max_steps required
                </Badge>
              )}
              <p className="text-xs text-muted">
                Model: <span className="font-mono text-fg">{model.data?.model_name ?? "not configured"}</span>
              </p>
              <ErrorNotice error={cfgQ.error} title="Could not load configuration" />
              <ErrorNotice error={generalError} title="Server rejected the request" />
              {Object.keys(serverErrors).length > 0 && (
                <p role="alert" className="text-xs text-err">
                  Fix the highlighted fields.
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                <Button type="submit" variant="secondary" loading={saveCfg.isPending}>
                  <Save aria-hidden /> Save config
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  onClick={onStart}
                  loading={start.isPending}
                  disabled={!selectedId || !!active}
                  title={active ? "A training job is already active" : !selectedId ? "Select a dataset first" : undefined}
                >
                  <Play aria-hidden /> Start training
                </Button>
              </div>
            </CardBody>
          </Card>
        </div>
      </form>
    </>
  );
}
