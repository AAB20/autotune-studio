/** Form primitives: Input, Label (Radix), Field wrapper with hint + inline error, Switch (Radix). */
import { forwardRef, useId, type InputHTMLAttributes, type ReactNode } from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import { cn } from "./cn";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-8 w-full min-w-0 rounded-md border border-edge bg-sunken px-2.5 text-[13px] text-fg placeholder:text-faint",
        "focus-visible:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30",
        "disabled:opacity-50 aria-[invalid=true]:border-err file:mr-2 file:border-0 file:bg-transparent file:text-muted",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export function Label({ className, ...props }: LabelPrimitive.LabelProps) {
  return <LabelPrimitive.Root className={cn("text-xs font-medium text-muted", className)} {...props} />;
}

interface FieldProps {
  label: ReactNode;
  /** Render prop receives the generated id + aria props to spread on the control. */
  children: (p: { id: string; "aria-invalid"?: boolean; "aria-describedby"?: string }) => ReactNode;
  hint?: ReactNode;
  error?: string | null;
  className?: string;
}

/** Labelled control with optional hint and inline error (wired via aria-describedby). */
export function Field({ label, children, hint, error, className }: FieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errId = `${id}-err`;
  const describedBy = [hint ? hintId : null, error ? errId : null].filter(Boolean).join(" ") || undefined;
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <Label htmlFor={id}>{label}</Label>
      {children({ id, "aria-invalid": error ? true : undefined, "aria-describedby": describedBy })}
      {hint && !error && (
        <p id={hintId} className="text-[11px] text-faint">
          {hint}
        </p>
      )}
      {error && (
        <p id={errId} role="alert" className="text-[11px] text-err">
          {error}
        </p>
      )}
    </div>
  );
}

interface SwitchProps {
  id?: string;
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}

export function Switch({ id, checked, onCheckedChange, label, description, disabled }: SwitchProps) {
  const gen = useId();
  const sid = id ?? gen;
  return (
    <div className="flex items-start gap-2.5">
      <SwitchPrimitive.Root
        id={sid}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
        className={cn(
          "relative mt-0.5 inline-flex h-[18px] w-8 shrink-0 cursor-pointer items-center rounded-full border border-edge-strong transition-colors",
          "bg-raised data-[state=checked]:border-accent data-[state=checked]:bg-accent/80 disabled:opacity-50",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        )}
      >
        <SwitchPrimitive.Thumb className="block size-3 translate-x-[2px] rounded-full bg-fg transition-transform data-[state=checked]:translate-x-[15px] data-[state=checked]:bg-accent-fg" />
      </SwitchPrimitive.Root>
      <div className="min-w-0">
        <Label htmlFor={sid} className="cursor-pointer text-[13px] text-fg">
          {label}
        </Label>
        {description && <p className="text-[11px] text-faint">{description}</p>}
      </div>
    </div>
  );
}
