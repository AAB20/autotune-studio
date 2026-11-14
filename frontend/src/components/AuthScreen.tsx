/** Token entry screen shown when no valid session cookie exists (ADR-0005). */
import { useState, type FormEvent } from "react";
import { KeyRound } from "lucide-react";
import { Button } from "./ui/button";
import { Field, Input } from "./ui/form";

interface AuthScreenProps {
  onSubmit: (token: string) => Promise<void>;
  error?: string | null;
  busy?: boolean;
}

export function AuthScreen({ onSubmit, error, busy }: AuthScreenProps) {
  const [token, setToken] = useState("");
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const t = token.trim();
    if (!t) return;
    await onSubmit(t);
    setToken("");
  };
  return (
    <div className="studio-grid flex h-full items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-lg border border-edge bg-panel p-5 shadow-2xl shadow-black/50">
        <div className="mb-4 flex items-center gap-2">
          <KeyRound className="size-4 text-accent" aria-hidden />
          <h1 className="text-[15px] font-semibold">AutoTune Studio</h1>
        </div>
        <p className="mb-4 text-xs text-muted">
          This session is not authenticated. Open the link printed by the notebook / CLI, or paste its access token below.
        </p>
        <Field label="Access token" error={error}>
          {(p) => (
            <Input
              {...p}
              type="password"
              autoComplete="off"
              autoFocus
              spellCheck={false}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste token"
            />
          )}
        </Field>
        <Button type="submit" variant="primary" className="mt-4 w-full" loading={busy} disabled={!token.trim()}>
          Unlock
        </Button>
      </form>
    </div>
  );
}
