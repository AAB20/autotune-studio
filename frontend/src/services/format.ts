/** Display formatters shared by all pages. Every function tolerates null/undefined and returns "—". */

export const DASH = "—";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** 1536 -> "1.5 KiB" (binary units, as GPU/RAM tools report them). */
export function formatBytes(bytes: number | null | undefined, digits = 1): string {
  if (!isNum(bytes) || bytes < 0) return DASH;
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i === 0 ? v.toFixed(0) : v.toFixed(digits)} ${units[i]}`;
}

/** Seconds -> "1h 02m", "3m 05s", "42s". null -> "—". */
export function formatDuration(seconds: number | null | undefined): string {
  if (!isNum(seconds) || seconds < 0) return DASH;
  const s = Math.round(seconds);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const p2 = (n: number) => String(n).padStart(2, "0");
  if (d > 0) return `${d}d ${p2(h)}h`;
  if (h > 0) return `${h}h ${p2(m)}m`;
  if (m > 0) return `${m}m ${p2(sec)}s`;
  return `${sec}s`;
}

/** Grouped integer / fixed decimal. */
export function formatNumber(v: number | null | undefined, digits = 0): string {
  if (!isNum(v)) return DASH;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Compact notation for learning rates / small floats: 0.0002 -> "2e-4", 0.05 -> "0.05". */
export function formatSci(v: number | null | undefined, digits = 2): string {
  if (!isNum(v)) return DASH;
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 0.01 && abs < 10000) return String(Number(v.toPrecision(digits + 1)));
  const [m = "0", e = "0"] = v.toExponential(digits).split("e");
  return `${Number(m)}e${Number(e)}`;
}

/** Loss-like float with 4 decimals. */
export function formatLoss(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return v.toFixed(4);
}

export function formatPct(v: number | null | undefined, digits = 0): string {
  if (!isNum(v)) return DASH;
  return `${v.toFixed(digits)}%`;
}

/** used/total -> percent in [0,100], or null when not computable. */
export function ratioPct(used: number | null | undefined, total: number | null | undefined): number | null {
  if (!isNum(used) || !isNum(total) || total <= 0) return null;
  return Math.max(0, Math.min(100, (used / total) * 100));
}

const p2 = (n: number) => String(n).padStart(2, "0");

/** Unix epoch seconds -> local "YYYY-MM-DD HH:MM". */
export function formatTimestamp(epochSeconds: number | null | undefined): string {
  if (!isNum(epochSeconds)) return DASH;
  const d = new Date(epochSeconds * 1000);
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}`;
}

/** Unix epoch seconds -> "HH:MM:SS" for log lines. */
export function formatClock(epochSeconds: number | null | undefined): string {
  if (!isNum(epochSeconds)) return DASH;
  const d = new Date(epochSeconds * 1000);
  return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
}

/** Render any JSON cell value as a single-line string, truncated for tables. */
export function formatCell(v: unknown, max = 160): string {
  let s: string;
  if (v === null || v === undefined) s = "null";
  else if (typeof v === "string") s = v;
  else {
    try {
      s = JSON.stringify(v);
    } catch {
      s = String(v);
    }
  }
  s = s.replace(/\s+/g, " ");
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}
