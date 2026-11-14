/** TanStack Query hooks, one per REST resource. Keys are centralised in `qk`. */
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, isApiRequestError } from "../services/api";
import { useEventStore } from "../services/eventStore";
import type { ModelConfig, TrainingStatus } from "../services/types";

export const qk = {
  env: ["env"] as const,
  hardware: ["hardware"] as const,
  resources: ["resources"] as const,
  sessions: ["sessions"] as const,
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["datasets", id] as const,
  preview: (id: string) => ["datasets", id, "preview"] as const,
  stats: (id: string) => ["datasets", id, "stats"] as const,
  model: ["model"] as const,
  suggested: ["models", "suggested"] as const,
  trainingConfig: ["training", "config"] as const,
  trainingStatus: ["training", "status"] as const,
  metrics: ["training", "metrics"] as const,
  checkpoints: ["checkpoints"] as const,
};

export const useEnv = () => useQuery({ queryKey: qk.env, queryFn: api.env, staleTime: 60_000 });
export const useHardware = () => useQuery({ queryKey: qk.hardware, queryFn: api.hardware, staleTime: 60_000 });
export const useSessions = () => useQuery({ queryKey: qk.sessions, queryFn: api.sessions });
export const useDatasets = () => useQuery({ queryKey: qk.datasets, queryFn: api.listDatasets });
export const useCheckpoints = () => useQuery({ queryKey: qk.checkpoints, queryFn: api.checkpoints });
export const useSuggestedModels = () =>
  useQuery({ queryKey: qk.suggested, queryFn: api.suggestedModels, staleTime: Infinity });
export const useTrainingConfig = () => useQuery({ queryKey: qk.trainingConfig, queryFn: api.getTrainingConfig });

/** Current model config, or null when none has been saved yet (404). */
export const useModelConfig = () =>
  useQuery<ModelConfig | null>({
    queryKey: qk.model,
    queryFn: async () => {
      try {
        return (await api.getModel()) ?? null;
      } catch (e) {
        if (isApiRequestError(e) && e.status === 404) return null;
        throw e;
      }
    },
  });

/** Latest resource snapshot: live WS value, seeded/polled from REST while the socket is down. */
export function useResources() {
  const live = useEventStore((s) => s.resources);
  const setResources = useEventStore((s) => s.setResources);
  const q = useQuery({ queryKey: qk.resources, queryFn: api.resources, refetchInterval: live ? false : 5000 });
  useEffect(() => {
    if (q.data) setResources(q.data);
  }, [q.data, setResources]);
  return live ?? q.data ?? null;
}

/** Training status: REST seeds the event store; WS `status` events keep it live. */
export function useTrainingStatus(): TrainingStatus | null {
  const live = useEventStore((s) => s.status);
  const setStatus = useEventStore((s) => s.setStatus);
  const q = useQuery({ queryKey: qk.trainingStatus, queryFn: api.trainingStatus });
  useEffect(() => {
    if (q.data) setStatus(q.data, q.dataUpdatedAt);
  }, [q.data, q.dataUpdatedAt, setStatus]);
  return live ?? q.data ?? null;
}

/** Metric series: REST history merged with live WS points. */
export function useMetrics() {
  const series = useEventStore((s) => s.metrics);
  const seed = useEventStore((s) => s.seedMetrics);
  const q = useQuery({ queryKey: qk.metrics, queryFn: api.trainingMetrics });
  useEffect(() => {
    if (q.data && Array.isArray(q.data)) seed(q.data);
  }, [q.data, seed]);
  return series;
}
