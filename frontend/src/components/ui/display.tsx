/** Display primitives: Badge, Meter (Radix Progress), Table, EmptyState, Segmented (Radix Tabs). */
import type { HTMLAttributes, ReactNode, TdHTMLAttributes, ThHTMLAttributes } from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "./cn";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium leading-none whitespace-nowrap [&_svg]:size-3",
  {
    variants: {
      tone: {
        neutral: "bg-raised text-muted ring-1 ring-inset ring-edge",
        accent: "bg-accent/12 text-accent ring-1 ring-inset ring-accent/30",
        ok: "bg-ok/10 text-ok ring-1 ring-inset ring-ok/30",
        warn: "bg-warn/10 text-warn ring-1 ring-inset ring-warn/30",
        err: "bg-err/10 text-err ring-1 ring-inset ring-err/30",
        info: "bg-info/10 text-info ring-1 ring-inset ring-info/30",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type BadgeTone = NonNullable<VariantProps<typeof badgeVariants>["tone"]>;

export function Badge({
  tone,
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

interface MeterProps {
  label: string;
  /** 0..100, or null when unknown (renders an empty track). */
  value: number | null;
  /** Right-aligned text, e.g. "6.1 / 15.0 GiB". */
  detail?: ReactNode;
  className?: string;
  /** Colour thresholds: warn above 80%, err above 92% unless `tone` is fixed. */
  tone?: "auto" | "accent" | "info";
}

/** Labelled horizontal usage bar (accessible progressbar). */
export function Meter({ label, value, detail, className, tone = "auto" }: MeterProps) {
  const v = value === null ? null : Math.max(0, Math.min(100, value));
  const color =
    tone === "accent"
      ? "bg-accent"
      : tone === "info"
        ? "bg-info"
        : v === null
          ? "bg-edge"
          : v > 92
            ? "bg-err"
            : v > 80
              ? "bg-warn"
              : "bg-ok";
  return (
    <div className={cn("min-w-0", className)}>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="text-muted">{label}</span>
        <span className="truncate font-mono text-fg">
          {detail}
          {v !== null && <span className="ml-1.5 text-faint">{v.toFixed(0)}%</span>}
        </span>
      </div>
      <ProgressPrimitive.Root
        aria-label={label}
        value={v}
        max={100}
        className="relative h-1.5 w-full overflow-hidden rounded-full bg-sunken ring-1 ring-inset ring-edge"
      >
        <ProgressPrimitive.Indicator
          className={cn("h-full rounded-full transition-[width] duration-500", color)}
          style={{ width: `${v ?? 0}%` }}
        />
      </ProgressPrimitive.Root>
    </div>
  );
}

export function Table({ className, ...props }: HTMLAttributes<HTMLTableElement>) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn("w-full border-collapse text-left text-[12.5px]", className)} {...props} />
    </div>
  );
}

export function Th({ className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      scope="col"
      className={cn(
        "sticky top-0 border-b border-edge bg-panel px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-faint",
        className,
      )}
      {...props}
    />
  );
}

export function Td({ className, ...props }: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn("border-b border-edge/60 px-3 py-1.5 align-top text-fg", className)} {...props} />;
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 px-4 py-8 text-center">
      {icon && <div className="text-faint [&_svg]:size-5">{icon}</div>}
      <p className="text-[13px] text-muted">{title}</p>
      {children && <div className="text-xs text-faint">{children}</div>}
    </div>
  );
}

interface SegmentedProps<T extends string> {
  value: T;
  onValueChange: (v: T) => void;
  options: { value: T; label: string }[];
  "aria-label": string;
}

/** Segmented control built on Radix Tabs (roving focus + arrow-key navigation). */
export function Segmented<T extends string>({ value, onValueChange, options, ...aria }: SegmentedProps<T>) {
  return (
    <TabsPrimitive.Root value={value} onValueChange={(v) => onValueChange(v as T)} activationMode="automatic">
      <TabsPrimitive.List
        aria-label={aria["aria-label"]}
        className="inline-flex flex-wrap gap-0.5 rounded-md border border-edge bg-sunken p-0.5"
      >
        {options.map((o) => (
          <TabsPrimitive.Trigger
            key={o.value}
            value={o.value}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium text-muted transition-colors hover:text-fg",
              "data-[state=active]:bg-raised data-[state=active]:text-fg data-[state=active]:shadow-[inset_0_0_0_1px_var(--color-edge-strong)]",
              "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent",
            )}
          >
            {o.label}
          </TabsPrimitive.Trigger>
        ))}
      </TabsPrimitive.List>
    </TabsPrimitive.Root>
  );
}
