/**
 * `useEvents()` – owns the single `/ws` connection for the app.
 *
 * Mount once (in the authenticated layout). Reconnect/backoff lives in `services/reconnect.ts`;
 * this hook wires it to React: connection state -> UI store, events -> event store reducer,
 * side effects (query invalidation, toasts) -> TanStack Query / toast store.
 */
import { useEffect } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { qk } from "./queries";
import { useEventStore } from "../services/eventStore";
import { ReconnectingEventStream, buildWsUrl, type ConnectionState, type SocketLike } from "../services/reconnect";
import { useUiStore } from "../services/uiStore";
import type { Event as StudioEvent, StreamStats } from "../services/types";

const TERMINAL = new Set(["completed", "stopped", "failed"]);

/** Side effects of an event beyond the store reducer. Exported for tests. */
export function handleEventSideEffects(ev: StudioEvent, qc: QueryClient): void {
  const data = (ev.data ?? {}) as Record<string, unknown>;
  switch (ev.type) {
    case "status":
      qc.setQueryData(qk.trainingStatus, data);
      if (typeof data.state === "string" && TERMINAL.has(data.state)) {
        void qc.invalidateQueries({ queryKey: qk.sessions });
        void qc.invalidateQueries({ queryKey: qk.checkpoints });
      }
      break;
    case "dataset": {
      const id = typeof data.dataset_id === "string" ? data.dataset_id : null;
      if (id) {
        const stats = (typeof data.stats === "object" && data.stats ? data.stats : data) as StreamStats;
        qc.setQueryData(qk.stats(id), stats);
        if (stats.state === "exhausted") {
          // A completed scan updates the dataset's cardinality: refresh the detail and the list.
          void qc.invalidateQueries({ queryKey: qk.dataset(id), exact: true });
          void qc.invalidateQueries({ queryKey: qk.datasets, exact: true });
        }
      }
      break;
    }
    case "checkpoint":
    case "export":
      void qc.invalidateQueries({ queryKey: qk.checkpoints });
      break;
    case "error":
      useUiStore.getState().toast({
        kind: "error",
        title: typeof data.code === "string" ? `Server error: ${data.code}` : "Server error",
        message: typeof data.message === "string" ? data.message : undefined,
      });
      break;
    default:
      break;
  }
}

export interface UseEventsOptions {
  /** Override for tests. */
  createSocket?: (url: string) => SocketLike;
  enabled?: boolean;
}

/** Connects to the event stream and returns the live connection state. */
export function useEvents(opts: UseEventsOptions = {}): ConnectionState {
  const qc = useQueryClient();
  const connection = useUiStore((s) => s.connection);
  const enabled = opts.enabled ?? true;
  const createSocket = opts.createSocket;

  useEffect(() => {
    if (!enabled) return;
    const { setConnection } = useUiStore.getState();
    const stream = new ReconnectingEventStream({
      url: (since) => buildWsUrl(window.location, since),
      createSocket: createSocket ?? ((url) => new WebSocket(url) as unknown as SocketLike),
      isOnline: () => (typeof navigator === "undefined" ? true : navigator.onLine !== false),
      onStateChange: setConnection,
      onEvent: (ev) => {
        useEventStore.getState().apply(ev);
        handleEventSideEffects(ev, qc);
      },
      onResync: () => {
        void qc.invalidateQueries();
      },
    });
    setConnection(stream.state);
    stream.start();
    const onOnline = () => stream.reconnectNow();
    const onVisible = () => {
      if (document.visibilityState === "visible") stream.reconnectNow();
    };
    window.addEventListener("online", onOnline);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("online", onOnline);
      document.removeEventListener("visibilitychange", onVisible);
      stream.stop();
    };
  }, [enabled, createSocket, qc]);

  return connection;
}
