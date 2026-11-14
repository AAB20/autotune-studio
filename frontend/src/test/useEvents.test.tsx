import { beforeEach, describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useEvents } from "../hooks/useEvents";
import { qk } from "../hooks/queries";
import { useEventStore } from "../services/eventStore";
import { useUiStore } from "../services/uiStore";
import type { SocketLike } from "../services/reconnect";

class FakeSocket implements SocketLike {
  onopen: SocketLike["onopen"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  onmessage: SocketLike["onmessage"] = null;
  closed = false;
  constructor(public url: string) {}
  close() {
    this.closed = true;
  }
}

beforeEach(() => {
  useEventStore.getState().reset();
  useUiStore.setState({ connection: "connecting", toasts: [] });
});

describe("useEvents", () => {
  it("connects to /ws?since=0, tracks state, dispatches events and side effects", () => {
    const sockets: FakeSocket[] = [];
    const createSocket = (url: string) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    };
    const qc = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
    const { result, unmount } = renderHook(() => useEvents({ createSocket }), { wrapper });

    expect(sockets[0]?.url).toBe("ws://localhost:3000/ws?since=0");
    expect(result.current).toBe("connecting");
    act(() => sockets[0]!.onopen?.({}));
    expect(result.current).toBe("open");

    const send = (o: unknown) => act(() => sockets[0]!.onmessage?.({ data: JSON.stringify(o) }));
    send({ type: "hello", data: { last_seq: 0, gap: false } });
    send({ seq: 1, type: "status", data: { state: "running", step: 3 } });
    send({ seq: 2, type: "metric", data: { step: 3, loss: 0.9 } });
    send({ seq: 3, type: "error", data: { code: "out_of_memory", message: "CUDA OOM" } });
    send({ seq: 4, type: "dataset", data: { dataset_id: "d", stats: { state: "streaming", rows_seen: 5 } } });

    expect(useEventStore.getState().status?.step).toBe(3);
    expect(qc.getQueryData(qk.trainingStatus)).toMatchObject({ state: "running" });
    expect(useEventStore.getState().metrics).toEqual([{ step: 3, loss: 0.9 }]);
    expect(useUiStore.getState().toasts.at(-1)).toMatchObject({ kind: "error", message: "CUDA OOM" });
    expect(qc.getQueryData(qk.stats("d"))).toMatchObject({ rows_seen: 5 });

    act(() => sockets[0]!.onclose?.({ code: 1006 }));
    expect(result.current).toBe("reconnecting");

    unmount();
    expect(sockets.every((s) => s.closed || s === sockets[0])).toBe(true);
  });
});
