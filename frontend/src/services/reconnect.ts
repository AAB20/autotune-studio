/**
 * Framework-free WebSocket reconnect logic for the `/ws` event stream.
 *
 * Protocol (fastapi-backend skill): connect `/ws?since=<lastSeq>`; the server first sends
 * `{"type":"hello","data":{"last_seq":N,"gap":bool}}`, then replayed + live `Event` JSON.
 * `{"type":"ping"}` frames are ignored. Everything here is injectable so it is unit-testable
 * with fake timers and a fake socket.
 */
import type { Event as StudioEvent } from "./types";

export type ConnectionState = "connecting" | "open" | "reconnecting" | "offline";

export interface BackoffOptions {
  /** First delay in ms (default 500). */
  baseMs?: number;
  /** Upper bound in ms (default 10 000). */
  maxMs?: number;
  /** Random source in [0,1) (default Math.random). */
  random?: () => number;
}

/**
 * Exponential backoff with "equal jitter": the exponential ceiling `min(max, base * 2^attempt)`
 * is randomised into `[ceiling/2, ceiling]`, and never below `base` nor above `max`.
 * attempt 0 -> 500 ms, attempt 1 -> 500..1000 ms, ... capped at 5..10 s.
 */
export function backoffDelay(attempt: number, opts: BackoffOptions = {}): number {
  const base = opts.baseMs ?? 500;
  const max = opts.maxMs ?? 10_000;
  const random = opts.random ?? Math.random;
  const ceiling = Math.min(max, base * 2 ** Math.max(0, Math.min(attempt, 30)));
  const jittered = ceiling / 2 + random() * (ceiling / 2);
  return Math.round(Math.min(max, Math.max(base, jittered)));
}

/** Build the WebSocket URL relative to the SPA's base path, e.g. wss://host/proxy/7860/ws?since=12. */
export function buildWsUrl(loc: Pick<Location, "protocol" | "host" | "pathname">, since: number): string {
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  let dir = loc.pathname || "/";
  if (!dir.endsWith("/")) {
    const last = dir.slice(dir.lastIndexOf("/") + 1);
    dir = last.includes(".") ? dir.slice(0, dir.lastIndexOf("/") + 1) : `${dir}/`;
  }
  return `${proto}//${loc.host}${dir}ws?since=${Math.max(0, Math.floor(since))}`;
}

/** Minimal subset of the browser WebSocket used here (so tests can provide a fake). */
export interface SocketLike {
  onopen: ((ev: unknown) => void) | null;
  onclose: ((ev: { code?: number; reason?: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  close(code?: number, reason?: string): void;
}

export interface HelloData {
  last_seq: number;
  gap: boolean;
}

export interface ReconnectingStreamOptions {
  /** Returns the URL for a given `since` sequence number. */
  url: (since: number) => string;
  createSocket: (url: string) => SocketLike;
  onEvent: (ev: StudioEvent) => void;
  onStateChange?: (state: ConnectionState) => void;
  /** Called when the client may have missed events (server gap or server restart): refetch REST state. */
  onResync?: (reason: "gap" | "server_reset") => void;
  onHello?: (hello: HelloData) => void;
  backoff?: BackoffOptions;
  /** Consecutive failed attempts after which the state is shown as "offline" (retries continue). */
  offlineAfter?: number;
  /** `navigator.onLine`-style probe; false short-circuits to "offline". */
  isOnline?: () => boolean;
  setTimer?: (fn: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
}

/** A WebSocket that reconnects forever with backoff and resumes from the last seen `seq`. */
export class ReconnectingEventStream {
  private readonly o: Required<Omit<ReconnectingStreamOptions, "onResync" | "onHello" | "onStateChange">> &
    Pick<ReconnectingStreamOptions, "onResync" | "onHello" | "onStateChange">;
  private socket: SocketLike | null = null;
  private timer: unknown = null;
  private attempt = 0;
  private stopped = true;
  private everOpened = false;
  private _state: ConnectionState = "connecting";
  private _lastSeq = 0;

  constructor(opts: ReconnectingStreamOptions) {
    // Drop explicit `undefined`s so they do not override the defaults below.
    const given = Object.fromEntries(Object.entries(opts).filter(([, v]) => v !== undefined)) as ReconnectingStreamOptions;
    this.o = {
      backoff: {},
      offlineAfter: 5,
      isOnline: () => true,
      setTimer: (fn, ms) => setTimeout(fn, ms),
      clearTimer: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
      ...given,
    };
  }

  get state(): ConnectionState {
    return this._state;
  }

  /** Highest event `seq` processed so far (sent as `since` on reconnect). */
  get lastSeq(): number {
    return this._lastSeq;
  }

  /** Number of consecutive failed connection attempts. */
  get attempts(): number {
    return this.attempt;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  /** Close permanently (component unmount). */
  stop(): void {
    this.stopped = true;
    this.clearPending();
    const s = this.socket;
    this.socket = null;
    if (s) {
      s.onopen = s.onclose = s.onerror = s.onmessage = null;
      try {
        s.close(1000, "client closing");
      } catch {
        /* ignore */
      }
    }
  }

  /** Skip the backoff wait and try now (e.g. browser `online` event or user click). */
  reconnectNow(): void {
    if (this.stopped || this._state === "open") return;
    this.clearPending();
    this.connect();
  }

  private setState(s: ConnectionState): void {
    if (this._state === s) return;
    this._state = s;
    this.o.onStateChange?.(s);
  }

  private clearPending(): void {
    if (this.timer !== null) {
      this.o.clearTimer(this.timer);
      this.timer = null;
    }
  }

  private connect(): void {
    if (this.stopped) return;
    if (!this.o.isOnline()) {
      this.setState("offline");
      this.scheduleRetry();
      return;
    }
    if (!this.everOpened && this.attempt === 0) this.setState("connecting");
    let socket: SocketLike;
    try {
      socket = this.o.createSocket(this.o.url(this._lastSeq));
    } catch {
      this.handleClose();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.attempt = 0;
      this.everOpened = true;
      this.setState("open");
    };
    socket.onmessage = (ev) => {
      if (this.socket === socket) this.handleMessage(ev.data);
    };
    socket.onerror = () => {
      /* onclose always follows; handled there */
    };
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.handleClose();
    };
  }

  private handleClose(): void {
    if (this.stopped) return;
    this.attempt += 1;
    const offline = !this.o.isOnline() || this.attempt >= this.o.offlineAfter;
    this.setState(offline ? "offline" : "reconnecting");
    this.scheduleRetry();
  }

  private scheduleRetry(): void {
    this.clearPending();
    const delay = backoffDelay(Math.max(0, this.attempt - 1), this.o.backoff);
    this.timer = this.o.setTimer(() => {
      this.timer = null;
      this.connect();
    }, delay);
  }

  /** Parse and route one frame. Exposed for tests via the socket's onmessage. */
  private handleMessage(raw: unknown): void {
    if (typeof raw !== "string") return;
    let msg: unknown;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (typeof msg !== "object" || msg === null) return;
    const m = msg as { type?: unknown; seq?: unknown; data?: unknown };
    if (m.type === "ping") return;
    if (m.type === "hello") {
      const d = (m.data ?? {}) as Partial<HelloData>;
      const hello: HelloData = { last_seq: Number(d.last_seq ?? 0) || 0, gap: Boolean(d.gap) };
      this.o.onHello?.(hello);
      if (hello.last_seq < this._lastSeq) {
        // Server restarted (sequence reset): accept its numbering and refetch everything.
        this._lastSeq = hello.last_seq;
        this.o.onResync?.("server_reset");
      } else if (hello.gap) {
        this.o.onResync?.("gap");
      }
      return;
    }
    if (typeof m.seq !== "number" || typeof m.type !== "string") return;
    if (m.seq <= this._lastSeq) return; // duplicate from replay
    this._lastSeq = m.seq;
    this.o.onEvent(msg as StudioEvent);
  }
}
