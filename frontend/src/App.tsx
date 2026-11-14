/** Root component: auth gate (ADR-0005) + hash router + query client. */
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Navigate, Route, Routes } from "react-router";
import { Loader2, ServerCrash } from "lucide-react";
import { Layout } from "./components/Layout";
import { AuthScreen } from "./components/AuthScreen";
import { Button } from "./components/ui/button";
import { DashboardPage } from "./pages/Dashboard";
import { DatasetsPage } from "./pages/Datasets";
import { ModelPage } from "./pages/Model";
import { TrainingPage } from "./pages/Training";
import { MonitorPage } from "./pages/Monitor";
import { CheckpointsPage } from "./pages/Checkpoints";
import { bootstrapAuth, checkSession, loginWithToken, type AuthState } from "./services/auth";
import { isApiRequestError } from "./services/api";
import { qk } from "./hooks/queries";

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5_000,
        refetchOnWindowFocus: false,
        retry: (count, err) => {
          // Never retry client errors (401/404/422...), retry network/5xx a couple of times.
          if (isApiRequestError(err) && err.status >= 400 && err.status < 500) return false;
          return count < 2;
        },
      },
      mutations: { retry: false },
    },
  });
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="model" element={<ModelPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="monitor" element={<MonitorPage />} />
        <Route path="checkpoints" element={<CheckpointsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function Splash({ children }: { children: ReactNode }) {
  return <div className="studio-grid flex h-full flex-col items-center justify-center gap-3 p-4 text-center text-muted">{children}</div>;
}

export default function App({ queryClient }: { queryClient?: QueryClient }) {
  const [qc] = useState(() => queryClient ?? createQueryClient());
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    void bootstrapAuth().then((s) => alive && setAuth(s));
    // A fresh launch link opened in an already-loaded tab only changes the hash: re-run the
    // token exchange (and drop cached data from the previous server session).
    const onHash = () => {
      if (!/(^#|[?&])token=/.test(window.location.hash)) return;
      void bootstrapAuth().then((s) => {
        if (!alive) return;
        qc.clear();
        setAuth(s);
      });
    };
    window.addEventListener("hashchange", onHash);
    return () => {
      alive = false;
      window.removeEventListener("hashchange", onHash);
    };
  }, [qc]);

  // Seed the env query so the layout does not refetch what auth already loaded.
  useEffect(() => {
    if (auth?.status === "authenticated") qc.setQueryData(qk.env, auth.env);
  }, [auth, qc]);

  const login = useCallback(async (token: string) => {
    setBusy(true);
    try {
      setAuth(await loginWithToken(token));
    } finally {
      setBusy(false);
    }
  }, []);

  const retry = useCallback(async () => {
    setBusy(true);
    try {
      setAuth(await checkSession());
    } finally {
      setBusy(false);
    }
  }, []);

  let body: ReactNode;
  if (auth === null) {
    body = (
      <Splash>
        <Loader2 className="size-5 animate-spin text-accent" aria-hidden />
        <p role="status">Connecting to AutoTune Studio…</p>
      </Splash>
    );
  } else if (auth.status === "unauthenticated") {
    body = <AuthScreen onSubmit={login} error={auth.reason ?? null} busy={busy} />;
  } else if (auth.status === "error") {
    body = (
      <Splash>
        <ServerCrash className="size-6 text-err" aria-hidden />
        <p role="alert" className="max-w-md">
          Cannot reach the studio server: {auth.reason}
        </p>
        <Button onClick={retry} loading={busy}>
          Retry
        </Button>
      </Splash>
    );
  } else {
    body = (
      <HashRouter>
        <AppRoutes />
      </HashRouter>
    );
  }

  return <QueryClientProvider client={qc}>{body}</QueryClientProvider>;
}
