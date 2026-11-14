/** RAM / VRAM / GPU-util / CPU meters from the latest ResourceSnapshot (falls back to HardwareInfo). */
import { Meter } from "./ui/display";
import { formatBytes, ratioPct } from "../services/format";
import type { HardwareInfo, ResourceSnapshot } from "../services/types";

interface Props {
  resources: ResourceSnapshot | null;
  hardware?: HardwareInfo | undefined;
  showCpu?: boolean;
}

export function ResourceMeters({ resources: r, hardware: hw, showCpu = true }: Props) {
  const gpu = hw?.gpus?.[0];
  const vramTotal = r?.vram_total_bytes ?? gpu?.total_vram_bytes ?? null;
  const vramUsed =
    r?.vram_used_bytes ?? (gpu && gpu.free_vram_bytes != null ? gpu.total_vram_bytes - gpu.free_vram_bytes : null);
  const ramTotal = r?.ram_total_bytes ?? hw?.ram_total_bytes ?? null;
  const ramUsed = r?.ram_used_bytes ?? (hw ? hw.ram_total_bytes - hw.ram_available_bytes : null);
  const gpuUtil = r?.gpu_util_pct ?? gpu?.utilization_pct ?? null;
  const hasGpu = vramTotal != null && vramTotal > 0;

  return (
    <div className="flex flex-col gap-3">
      {hasGpu ? (
        <>
          <Meter
            label="VRAM"
            value={ratioPct(vramUsed, vramTotal)}
            detail={`${formatBytes(vramUsed)} / ${formatBytes(vramTotal)}`}
          />
          {r?.vram_reserved_bytes != null && (
            <p className="-mt-2 text-[11px] text-faint">Reserved by allocator: {formatBytes(r.vram_reserved_bytes)}</p>
          )}
          <Meter label="GPU utilisation" value={gpuUtil} tone="info" detail={gpuUtil == null ? "n/a" : undefined} />
        </>
      ) : (
        <p className="text-xs text-faint">No CUDA GPU — VRAM metrics unavailable.</p>
      )}
      <Meter label="RAM" value={ratioPct(ramUsed, ramTotal)} detail={`${formatBytes(ramUsed)} / ${formatBytes(ramTotal)}`} />
      {showCpu && <Meter label="CPU" value={r?.cpu_pct ?? null} tone="info" detail={r ? undefined : "n/a"} />}
    </div>
  );
}
