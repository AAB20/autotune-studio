/** Small client-only UI state: connection status, selected dataset and toasts. */
import { create } from "zustand";
import type { ConnectionState } from "./reconnect";

export type ToastKind = "error" | "success" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  message?: string;
}

const SELECTED_KEY = "autotune.selectedDataset";

function readSelected(): string | null {
  try {
    return window.localStorage.getItem(SELECTED_KEY);
  } catch {
    return null;
  }
}

export interface UiStore {
  connection: ConnectionState;
  setConnection: (c: ConnectionState) => void;
  selectedDatasetId: string | null;
  selectDataset: (id: string | null) => void;
  toasts: Toast[];
  toast: (t: Omit<Toast, "id">) => void;
  dismiss: (id: number) => void;
}

let nextToastId = 1;

export const useUiStore = create<UiStore>()((set) => ({
  connection: "connecting",
  setConnection: (connection) => set({ connection }),
  selectedDatasetId: readSelected(),
  selectDataset: (id) => {
    try {
      if (id) window.localStorage.setItem(SELECTED_KEY, id);
      else window.localStorage.removeItem(SELECTED_KEY);
    } catch {
      /* storage unavailable (sandboxed iframe) */
    }
    set({ selectedDatasetId: id });
  },
  toasts: [],
  toast: (t) => set((s) => ({ toasts: [...s.toasts.slice(-4), { ...t, id: nextToastId++ }] })),
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

/** Convenience for non-React code paths. */
export const notify = (t: Omit<Toast, "id">) => useUiStore.getState().toast(t);
