/** Shared app-level components: connection/training badges, error notice, toaster, page header. */
import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { Badge, type BadgeTone } from "./ui/display";
import { isApiRequestError } from "../services/api";
import type { ConnectionState } from "../services/reconnect";
import type { RuntimeKind, TrainingState } from "../services/types";
import { useUiStore } from "../services/uiStore";
import { cn } from "./ui/cn";

const CONNECTION: Record<ConnectionState, { label: string; tone: BadgeTone; dot: string }> = {
  open: { label: "Connected", tone: "ok", dot: "bg-ok" },
  connecting: { label: "Connecting", tone: "info", dot: "bg-info pulse-dot" },
  reconnecting: { label: "Reconnecting", tone: "warn", dot: "bg-warn pulse-dot" },
  offline: { label: "Offline", tone: "err", dot: "bg-err" },
};

export function ConnectionBadge({ state }: { state: ConnectionState }) {
  const c = CONNECTION[state];
  return (
    <Badge tone={c.tone} role="status" aria-live="polite" aria-label={`Live connection: ${c.label}`}>
      <span className={cn("size-1.5 rounded-full", c.dot)} aria-hidden />
      {c.label}
    </Badge>
  );
}

const RUNTIME_LABEL: Record<RuntimeKind, string> = {
  colab: "Colab",
  kaggle: "Kaggle",
  jupyter: "Jupyter",
  script: "Script",
};

export function RuntimeBadge({ kind }: { kind: RuntimeKind | undefined }) {
  return <Badge tone="accent">{kind ? RUNTIME_LABEL[kind] : "Runtime ?"}</Badge>;
}

const STATE_TONE: Record<TrainingState, BadgeTone> = {
  idle: "neutral",
  preparing: "info",
  running: "ok",
  paused: "warn",
  stopping: "warn",
  completed: "accent",
  stopped: "neutral",
  failed: "err",
};

export function TrainingStateBadge({ state }: { state: TrainingState | undefined }) {
  const s = state ?? "idle";
  return (
    <Badge tone={STATE_TONE[s]} aria-label={`Training state: ${s}`}>
      {(s === "running" || s === "preparing") && <span className="size-1.5 rounded-full bg-current pulse-dot" aria-hidden />}
      {s}
    </Badge>
  );
}

/** Inline error block for a failed query/mutation. */
export function ErrorNotice({ error, title = "Request failed" }: { error: unknown; title?: string }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  const code = isApiRequestError(error) ? error.code : null;
  return (
    <div role="alert" className="flex items-start gap-2 rounded-md border border-err/30 bg-err/8 px-3 py-2 text-xs text-err">
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      <div className="min-w-0">
        <p className="font-medium">
          {title}
          {code && <span className="ml-1.5 font-mono text-[11px] text-err/70">{code}</span>}
        </p>
        <p className="break-words text-err/90">{msg}</p>
      </div>
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-lg font-semibold tracking-tight text-fg">{title}</h1>
        {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Toaster() {
  const toasts = useUiStore((s) => s.toasts);
  const dismiss = useUiStore((s) => s.dismiss);
  return (
    <div aria-live="assertive" className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((t) => {
        const Icon = t.kind === "error" ? AlertTriangle : t.kind === "success" ? CheckCircle2 : Info;
        const color = t.kind === "error" ? "text-err" : t.kind === "success" ? "text-ok" : "text-info";
        return (
          <div
            key={t.id}
            role={t.kind === "error" ? "alert" : "status"}
            className="pointer-events-auto flex items-start gap-2 rounded-md border border-edge-strong bg-raised p-3 shadow-xl shadow-black/40"
          >
            <Icon className={cn("mt-0.5 size-4 shrink-0", color)} aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-[13px] font-medium text-fg">{t.title}</p>
              {t.message && <p className="mt-0.5 break-words text-xs text-muted">{t.message}</p>}
            </div>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              className="rounded p-0.5 text-faint hover:text-fg"
              aria-label="Dismiss notification"
            >
              <X className="size-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
