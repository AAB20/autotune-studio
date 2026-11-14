/** Bounded, auto-scrolling log console fed by WS `log` / `error` events. */
import { useEffect, useRef, useState } from "react";
import { useEventStore, type LogLine } from "../services/eventStore";
import { formatClock } from "../services/format";
import { cn } from "./ui/cn";

const LEVEL_COLOR: Record<string, string> = {
  error: "text-err",
  critical: "text-err",
  warning: "text-warn",
  warn: "text-warn",
  info: "text-info",
  debug: "text-faint",
};

export function LogConsole({ className }: { className?: string }) {
  const logs = useEventStore((s) => s.logs);
  const ref = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  useEffect(() => {
    const el = ref.current;
    if (el && follow) el.scrollTop = el.scrollHeight;
  }, [logs, follow]);

  return (
    <div
      ref={ref}
      role="log"
      aria-label="Training log"
      tabIndex={0}
      onScroll={(e) => {
        const el = e.currentTarget;
        setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
      }}
      className={cn("h-64 overflow-y-auto bg-sunken px-3 py-2 font-mono text-[11.5px] leading-relaxed", className)}
    >
      {logs.length === 0 ? (
        <p className="text-faint">Waiting for log output…</p>
      ) : (
        logs.map((l: LogLine) => (
          <div key={l.seq} className="flex gap-2 whitespace-pre-wrap break-all">
            <span className="shrink-0 text-faint">{formatClock(l.ts)}</span>
            <span className={cn("w-12 shrink-0 uppercase", LEVEL_COLOR[l.level] ?? "text-muted")}>{l.level.slice(0, 5)}</span>
            <span className="text-fg/90">{l.message}</span>
          </div>
        ))
      )}
    </div>
  );
}
