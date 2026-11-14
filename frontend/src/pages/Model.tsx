/** Model selection: HF repo id (or local path), suggested models, max seq length, 4-bit, dtype. */
import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Boxes, ShieldCheck, Sparkles } from "lucide-react";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Field, Input, Switch } from "../components/ui/form";
import { Select } from "../components/ui/select";
import { EmptyState } from "../components/ui/display";
import { ErrorNotice, PageHeader } from "../components/common";
import { qk, useModelConfig, useSuggestedModels } from "../hooks/queries";
import { api, fieldErrors } from "../services/api";
import { useUiStore } from "../services/uiStore";
import type { ModelConfig } from "../services/types";

type Dtype = NonNullable<ModelConfig["dtype"]>;
const DTYPES: { value: Dtype; label: string }[] = [
  { value: "auto", label: "auto" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float16", label: "float16" },
  { value: "float32", label: "float32" },
];

export function ModelPage() {
  const qc = useQueryClient();
  const toast = useUiStore((s) => s.toast);
  const current = useModelConfig();
  const suggested = useSuggestedModels();
  const [name, setName] = useState("");
  const [revision, setRevision] = useState("");
  const [seqLen, setSeqLen] = useState("2048");
  const [fourBit, setFourBit] = useState(true);
  const [dtype, setDtype] = useState<Dtype>("auto");
  const [clientError, setClientError] = useState<string | null>(null);

  useEffect(() => {
    const c = current.data;
    if (!c) return;
    setName(c.model_name);
    setRevision(c.revision ?? "");
    setSeqLen(String(c.max_seq_length ?? 2048));
    setFourBit(c.load_in_4bit ?? true);
    setDtype(c.dtype ?? "auto");
  }, [current.data]);

  const save = useMutation({
    mutationFn: (cfg: ModelConfig) => api.putModel(cfg),
    onSuccess: (cfg) => {
      qc.setQueryData(qk.model, cfg);
      toast({ kind: "success", title: "Model configuration saved", message: cfg?.model_name });
    },
  });
  const errs = fieldErrors(save.error);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const n = Number(seqLen);
    if (!Number.isInteger(n) || n < 16 || n > 131072) {
      setClientError("Max sequence length must be an integer between 16 and 131072.");
      return;
    }
    setClientError(null);
    save.mutate({
      model_name: name.trim(),
      revision: revision.trim() || null,
      max_seq_length: n,
      load_in_4bit: fourBit,
      dtype,
    });
  };

  return (
    <>
      <PageHeader title="Model" description="Base model to fine-tune with LoRA / QLoRA." />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader title="Base model" icon={<Boxes />} />
          <CardBody>
            <form onSubmit={submit} className="flex flex-col gap-3.5">
              <Field label="Hugging Face model id or local path" error={errs.model_name} hint="e.g. unsloth/Llama-3.2-1B-Instruct">
                {(p) => <Input {...p} value={name} onChange={(e) => setName(e.target.value)} spellCheck={false} placeholder="org/model" required />}
              </Field>
              <div className="grid gap-3 sm:grid-cols-3">
                <Field label="Revision (optional)" error={errs.revision}>
                  {(p) => <Input {...p} value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="main" />}
                </Field>
                <Field label="Max sequence length" error={errs.max_seq_length ?? clientError}>
                  {(p) => <Input {...p} type="number" min={16} max={131072} step={1} inputMode="numeric" value={seqLen} onChange={(e) => setSeqLen(e.target.value)} />}
                </Field>
                <Field label="Compute dtype" error={errs.dtype}>
                  {(p) => <Select {...p} value={dtype} onValueChange={(v) => setDtype(v as Dtype)} options={DTYPES} />}
                </Field>
              </div>
              <Switch
                checked={fourBit}
                onCheckedChange={setFourBit}
                label="Load in 4-bit"
                description="bitsandbytes NF4 quantisation (QLoRA). Requires a CUDA GPU."
              />
              <div className="flex items-start gap-2 rounded-md border border-edge bg-sunken px-3 py-2 text-xs text-muted">
                <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-ok" aria-hidden />
                <p>
                  Remote code is disabled: <span className="font-mono">trust_remote_code=False</span> is enforced by the server.
                  Models that need custom modelling code cannot be loaded.
                </p>
              </div>
              <ErrorNotice error={save.error && Object.keys(errs).length === 0 ? save.error : null} title="Save failed" />
              <ErrorNotice error={current.error} title="Could not read current model" />
              <div>
                <Button type="submit" variant="primary" loading={save.isPending} disabled={!name.trim()}>
                  Save model
                </Button>
              </div>
            </form>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Suggested models" icon={<Sparkles />} subtitle="pre-quantised, Unsloth-compatible" />
          <ErrorNotice error={suggested.error} />
          {suggested.data && suggested.data.length > 0 ? (
            <ul className="max-h-[420px] overflow-y-auto p-1.5">
              {suggested.data.map((m) => (
                <li key={m}>
                  <button
                    type="button"
                    onClick={() => setName(m)}
                    aria-pressed={m === name}
                    className="w-full truncate rounded px-2 py-1.5 text-left font-mono text-xs text-muted hover:bg-raised hover:text-fg aria-[pressed=true]:bg-raised aria-[pressed=true]:text-accent"
                    title={m}
                  >
                    {m}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            !suggested.isLoading && <EmptyState title="No suggestions available" />
          )}
        </Card>
      </div>
    </>
  );
}
