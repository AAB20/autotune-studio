import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "./cn";

export function Card({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <section className={cn("rounded-lg border border-edge bg-panel", className)} {...props} />;
}

interface CardHeaderProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  subtitle?: ReactNode;
}

/** Compact header strip: small uppercase title, optional icon, subtitle and right-aligned actions. */
export function CardHeader({ title, icon, actions, subtitle, className, ...props }: CardHeaderProps) {
  return (
    <div className={cn("flex items-center gap-2 border-b border-edge px-3.5 py-2.5", className)} {...props}>
      {icon && <span className="text-faint [&_svg]:size-3.5">{icon}</span>}
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{title}</h2>
        {subtitle && <p className="truncate text-xs text-faint">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-3.5", className)} {...props} />;
}

/** Label/value pair used in dense stat grids. */
export function Stat({
  label,
  value,
  hint,
  mono = true,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <dt className="text-[11px] uppercase tracking-wider text-faint">{label}</dt>
      <dd className={cn("truncate text-[15px] text-fg", mono && "font-mono")}>{value}</dd>
      {hint && <dd className="truncate text-xs text-faint">{hint}</dd>}
    </div>
  );
}
