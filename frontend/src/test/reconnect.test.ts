import { describe, expect, it, vi } from "vitest";
import {
  ReconnectingEventStream,
  backoffDelay,
  buildWsUrl,
  type ConnectionState,
  type SocketLike,
} from "../services/reconnect";
import type { Event as StudioEvent } from "../services/types";

class FakeSocket implements SocketLike {
  onopen: SocketLike["onopen"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  onmessage: SocketLike["onmessage"] = null;
  closed = false;
  constructor(public url: string) {}
  open() {
    this.onopen?.({});
  }
  send(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  drop(code = 1006) {
    this.onclose?.({ code });
  }
  close() {
    this.closed = true;
  }
}

function setup(opts: { offlineAfter?: number; online?: () => boolean } = {}) {
  const sockets: FakeSocket[] = [];
  const timers: { fn: () => void; ms: number }[] = [];
  const states: ConnectionState[] = [];
  const events: StudioEvent[] = [];
  const onResync = vi.fn();
  const stream = new ReconnectingEventStream({
    url: (since) => `ws://h/ws?since=${since}`,
    createSocket: (url) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    },
    onEvent: (e) => events.push(e),
    onStateChange: (s) => states.push(s),
    onResync,
    backoff: { random: () => 1 },
    offlineAfter: opts.offlineAfter ?? 5,
    isOnline: opts.online,
    setTimer: (fn, ms) => {
      timers.push({ fn, ms });
      return timers.length;
    },
    clearTimer: () => {},
  });
  const fireTimer = () => {
    const t = timers.shift();
    if (!t) throw new Error("no timer pending");
    t.fn();
    return t.ms;
  };
  return { stream, sockets, timers, states, events, onResync, fireTimer };
}

describe("backoffDelay", () => {
  it("starts at 0.5s and caps at 10s", () => {
    expect(backoffDelay(0, { random: () => 0 })).toBe(500);
    expect(backoffDelay(0, { random: () => 0.999 })).toBe(500);
    expect(backoffDelay(1, { random: () => 1 })).toBe(1000);
    expect(backoffDelay(3, { random: () => 1 })).toBe(4000);
    expect(backoffDelay(10, { random: () => 1 })).toBe(10000);
    expect(backoffDelay(50, { random: () => 1 })).toBe(10000);
  });

  it("applies jitter within [ceiling/2, ceiling]", () => {
    for (let a = 0; a < 12; a++) {
      const ceiling = Math.min(10000, 500 * 2 ** a);
      for (const r of [0, 0.25, 0.5, 0.99]) {
        const d = backoffDelay(a, { random: () => r });
        expect(d).toBeGreaterThanOrEqual(Math.max(500, ceiling / 2));
        expect(d).toBeLessThanOrEqual(ceiling);
      }
    }
    expect(backoffDelay(4, { random: () => 0 })).toBe(4000);
    expect(backoffDelay(4, { random: () => 0.5 })).toBe(6000);
  });
});

describe("buildWsUrl", () => {
  it("derives ws/wss and keeps the proxy base path", () => {
    expect(buildWsUrl({ protocol: "http:", host: "127.0.0.1:7860", pathname: "/" }, 0)).toBe("ws://127.0.0.1:7860/ws?since=0");
    expect(buildWsUrl({ protocol: "https:", host: "x.colab.dev", pathname: "/proxy/7860/" }, 42)).toBe(
      "wss://x.colab.dev/proxy/7860/ws?since=42",
    );
    expect(buildWsUrl({ protocol: "https:", host: "k", pathname: "/proxy/7860/index.html" }, 3)).toBe("wss://k/proxy/7860/ws?since=3");
    expect(buildWsUrl({ protocol: "http:", host: "k", pathname: "/proxy/7860" }, 1)).toBe("ws://k/proxy/7860/ws?since=1");
  });
});

describe("ReconnectingEventStream", () => {
  it("connects, opens and dispatches events (dedupes replayed seqs, ignores ping)", () => {
    const t = setup();
    t.stream.start();
    expect(t.stream.state).toBe("connecting");
    expect(t.sockets[0]?.url).toBe("ws://h/ws?since=0");
    t.sockets[0]!.open();
    expect(t.stream.state).toBe("open");
    const s = t.sockets[0]!;
    s.send({ type: "hello", data: { last_seq: 2, gap: false } });
    s.send({ seq: 1, type: "log", data: { message: "a" } });
    s.send({ seq: 2, type: "log", data: { message: "b" } });
    s.send({ seq: 2, type: "log", data: { message: "dup" } });
    s.send({ type: "ping" });
    s.send("not json" as unknown);
    expect(t.events.map((e) => e.seq)).toEqual([1, 2]);
    expect(t.stream.lastSeq).toBe(2);
    expect(t.onResync).not.toHaveBeenCalled();
  });

  it("reconnects with since=lastSeq and exponential delays, then resets attempts on open", () => {
    const t = setup({ offlineAfter: 10 });
    t.stream.start();
    t.sockets[0]!.open();
    t.sockets[0]!.send({ seq: 7, type: "metric", data: { step: 1 } });
    t.sockets[0]!.drop();
    expect(t.stream.state).toBe("reconnecting");

    const delays: number[] = [];
    delays.push(t.fireTimer());
    expect(t.sockets[1]?.url).toBe("ws://h/ws?since=7");
    t.sockets[1]!.drop();
    delays.push(t.fireTimer());
    t.sockets[2]!.drop();
    delays.push(t.fireTimer());
    expect(delays).toEqual([500, 1000, 2000]);

    t.sockets[3]!.open();
    expect(t.stream.state).toBe("open");
    expect(t.stream.attempts).toBe(0);
    t.sockets[3]!.drop();
    expect(t.fireTimer()).toBe(500);
  });

  it("goes offline after N consecutive failures but keeps retrying (capped)", () => {
    const t = setup({ offlineAfter: 3 });
    t.stream.start();
    t.sockets[0]!.drop();
    expect(t.stream.state).toBe("reconnecting");
    t.fireTimer();
    t.sockets[1]!.drop();
    t.fireTimer();
    t.sockets[2]!.drop();
    expect(t.stream.state).toBe("offline");
    for (let i = 3; i < 12; i++) {
      const ms = t.fireTimer();
      expect(ms).toBeLessThanOrEqual(10000);
      t.sockets[i]!.drop();
    }
    expect(t.timers).toHaveLength(1);
    t.fireTimer();
    t.sockets[12]!.open();
    expect(t.stream.state).toBe("open");
  });

  it("reports offline immediately when the browser is offline", () => {
    let online = false;
    const t = setup({ online: () => online });
    t.stream.start();
    expect(t.stream.state).toBe("offline");
    expect(t.sockets).toHaveLength(0);
    online = true;
    t.fireTimer();
    expect(t.sockets).toHaveLength(1);
  });

  it("requests a resync on hello.gap and on server sequence reset", () => {
    const t = setup();
    t.stream.start();
    t.sockets[0]!.open();
    t.sockets[0]!.send({ type: "hello", data: { last_seq: 100, gap: true } });
    expect(t.onResync).toHaveBeenCalledWith("gap");
    t.sockets[0]!.send({ seq: 100, type: "status", data: {} });
    t.sockets[0]!.drop();
    t.fireTimer();
    // Server restarted: its seq counter is back to 3.
    t.sockets[1]!.open();
    t.sockets[1]!.send({ type: "hello", data: { last_seq: 3, gap: false } });
    expect(t.onResync).toHaveBeenLastCalledWith("server_reset");
    t.sockets[1]!.send({ seq: 4, type: "log", data: { message: "new" } });
    expect(t.events.at(-1)?.seq).toBe(4);
  });

  it("stop() closes the socket and cancels reconnects", () => {
    const t = setup();
    t.stream.start();
    t.sockets[0]!.open();
    t.stream.stop();
    expect(t.sockets[0]!.closed).toBe(true);
    t.sockets[0]!.drop();
    expect(t.timers).toHaveLength(0);
  });

  it("ignores events from a superseded socket", () => {
    const t = setup();
    t.stream.start();
    const old = t.sockets[0]!;
    old.drop();
    t.fireTimer();
    old.send({ seq: 5, type: "log", data: {} });
    expect(t.events).toHaveLength(0);
  });
});
