/** App shell: sidebar navigation + top bar (runtime, GPU, connection) + routed content. */
import { NavLink, Outlet } from "react-router";
import { Activity, Boxes, Cpu, Database, Gauge, HardDriveDownload, Settings, SlidersHorizontal } from "lucide-react";
import { ConnectionBadge, RuntimeBadge, Toaster, TrainingStateBadge } from "./common";
import { Badge } from "./ui/display";
import { cn } from "./ui/cn";
import { useEnv, useHardware, useTrainingStatus } from "../hooks/queries";
import { useEvents } from "../hooks/useEvents";

const NAV = [
  { to: "/", label: "Dashboard", icon: Gauge, end: true },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/model", label: "Model", icon: Boxes },
  { to: "/training", label: "Training", icon: SlidersHorizontal },
  { to: "/monitor", label: "Monitor", icon: Activity },
  { to: "/checkpoints", label: "Checkpoints", icon: HardDriveDownload },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Logo() {
  return (
    <div className="flex h-12 items-center gap-2 border-b border-edge px-3.5">
      <svg viewBox="0 0 24 24" className="size-5 shrink-0" aria-hidden>
        <path d="M2 17 L7 9 L11 13 L16 5 L22 11" fill="none" stroke="var(--color-accent)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="16" cy="5" r="2" fill="var(--color-accent)" />
      </svg>
      <span className="hidden text-[13px] font-semibold tracking-tight lg:inline">
        AutoTune <span className="text-muted">Studio</span>
      </span>
    </div>
  );
}

export function Layout() {
  const connection = useEvents();
  const env = useEnv();
  const hw = useHardware();
  const status = useTrainingStatus();
  const gpu = hw.data?.gpus?.[0];

  return (
    <div className="flex h-full min-h-0">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:rounded focus:bg-accent focus:px-2 focus:py-1 focus:text-accent-fg">
        Skip to content
      </a>
      <aside className="flex w-14 shrink-0 flex-col border-r border-edge bg-panel lg:w-52">
        <Logo />
        <nav aria-label="Main" className="flex flex-1 flex-col gap-0.5 p-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={label}
              className={({ isActive }) =>
                cn(
                  "group relative flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] text-muted transition-colors hover:bg-raised hover:text-fg",
                  isActive && "bg-raised text-fg before:absolute before:inset-y-1.5 before:left-0 before:w-0.5 before:rounded-full before:bg-accent",
                )
              }
            >
              <Icon className="size-4 shrink-0" aria-hidden />
              <span className="sr-only lg:not-sr-only">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="hidden border-t border-edge p-3 text-[11px] text-faint lg:block">
          Remote code execution disabled
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-edge bg-panel/80 px-4 backdrop-blur">
          <RuntimeBadge kind={env.data?.kind} />
          <Badge tone="neutral" className="max-w-[40vw]" title={gpu ? gpu.name : "No CUDA GPU detected"}>
            <Cpu aria-hidden />
            <span className="truncate">{hw.isLoading ? "Detecting…" : gpu ? gpu.name : "CPU only"}</span>
          </Badge>
          <div className="flex-1" />
          {status && status.state !== "idle" && <TrainingStateBadge state={status.state} />}
          <ConnectionBadge state={connection} />
        </header>
        <main id="main" className="studio-grid min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] p-4 md:p-5">
            <Outlet />
          </div>
        </main>
      </div>
      <Toaster />
    </div>
  );
}
