/** Settings: Hugging Face username + access token for pushing exports to the Hub (ADR-0007).
 *
 * The token is write-only: it is sent once to the server, verified there, kept in server memory only,
 * and never returned or stored in the browser. The input is cleared after saving. */
import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { KeyRound, ShieldCheck, Trash2 } from "lucide-react";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/form";
import { Badge } from "../components/ui/display";
import { ErrorNotice, PageHeader } from "../components/common";
import { qk, useHfCredentials } from "../hooks/queries";
import { api, fieldErrors } from "../services/api";
import { useUiStore } from "../services/uiStore";
import type { HFCredentialsStatus } from "../services/types";

export function HfStatusBadge({ status }: { status: HFCredentialsStatus | undefined }) {
  if (!status?.configured) return <Badge tone="neutral">not connected</Badge>;
  if (status.can_write === false) return <Badge tone="warn">read-only token</Badge>;
  return <Badge tone="ok">connected as {status.username}</Badge>;
}

export function SettingsPage() {
  const qc = useQueryClient();
  const toast = useUiStore((s) => s.toast);
  const status = useHfCredentials();
  const [username, setUsername] = useState("");
  const [token, setToken] = useState("");

  const save = useMutation({
    mutationFn: () => api.putHfCredentials({ username: username.trim(), token: token.trim() }),
    onSuccess: (st) => {
      setToken(""); // never keep the token in the page
      qc.setQueryData(qk.hfCredentials, st);
      toast({ kind: "success", title: "Hugging Face connected", message: st.username ?? undefined });
    },
    onError: () => setToken(""),
  });
  const clear = useMutation({
    mutationFn: api.deleteHfCredentials,
    onSuccess: (st) => {
      qc.setQueryData(qk.hfCredentials, st);
      toast({ kind: "info", title: "Hugging Face credentials removed" });
    },
  });
  const errs = fieldErrors(save.error);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    save.mutate();
  };
  const st = status.data;

  return (
    <>
      <PageHeader title="Settings" description="Accounts used by this studio session." />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader title="Hugging Face" icon={<KeyRound />} subtitle="used to push exported models to the Hub" />
          <CardBody>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-muted">
              <span>Status:</span>
              <HfStatusBadge status={st} />
              {st?.configured && st.token_role && <span className="font-mono">token role: {st.token_role}</span>}
            </div>
            {st?.configured && st.can_write === null && (
              <p className="mb-3 text-xs text-muted">
                Fine-grained token: write access is checked per repository when you push.
              </p>
            )}
            <form onSubmit={submit} className="flex flex-col gap-3.5">
              <Field label="Username or organisation" error={errs.username} hint="Models are pushed to <username>/<repo>.">
                {(p) => (
                  <Input
                    {...p}
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder={st?.username ?? "your-hf-username"}
                    autoComplete="username"
                    spellCheck={false}
                    required
                  />
                )}
              </Field>
              <Field
                label="Access token"
                error={errs.token}
                hint="Create one at huggingface.co/settings/tokens with Write access."
              >
                {(p) => (
                  <Input
                    {...p}
                    type="password"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    placeholder={st?.configured ? "saved (enter a new token to replace it)" : "hf_..."}
                    autoComplete="off"
                    spellCheck={false}
                    required
                  />
                )}
              </Field>
              <ErrorNotice error={save.error && Object.keys(errs).length === 0 ? save.error : null} title="Could not verify" />
              <ErrorNotice error={status.error} title="Could not read status" />
              <div className="flex gap-2">
                <Button type="submit" variant="primary" loading={save.isPending} disabled={!username.trim() || !token.trim()}>
                  Verify &amp; save
                </Button>
                {st?.configured && (
                  <Button type="button" variant="ghost" loading={clear.isPending} onClick={() => clear.mutate()}>
                    <Trash2 aria-hidden /> Remove
                  </Button>
                )}
              </div>
            </form>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="How the token is handled" icon={<ShieldCheck />} />
          <CardBody>
            <ul className="list-disc space-y-1.5 pl-4 text-xs text-muted">
              <li>Verified with the Hub, then kept in the server&apos;s memory for this session only.</li>
              <li>Never written to disk, never sent back to the browser, masked in logs.</li>
              <li>Lost when the notebook kernel restarts: enter it again after a restart.</li>
              <li>Used only for pushing exports. Gated model downloads use the HF_TOKEN notebook secret.</li>
            </ul>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
