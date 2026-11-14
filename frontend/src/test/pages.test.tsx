import { beforeEach, describe, expect, it } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { DatasetsPage } from "../pages/Datasets";
import { MonitorPage } from "../pages/Monitor";
import { TrainingPage } from "../pages/Training";
import { useEventStore } from "../services/eventStore";
import { useUiStore } from "../services/uiStore";
import type { DatasetInfo, StreamStats, TrainingStatus } from "../services/types";
import { mockFetch, renderWithProviders } from "./helpers";

const STREAMING_DS: DatasetInfo = {
  dataset_id: "ds-1",
  spec: { source: "huggingface", name: "org/huge-corpus", split: "train" },
  columns: [
    { name: "text", dtype: "string", sampled_null_count: 0, examples: ["hello world", "foo"] },
    { name: "meta", dtype: "struct", sampled_null_count: 3, examples: ["{}"] },
  ],
  cardinality: { rows: null, exact: false, method: "unknown" },
  detected_format: "text",
  text_columns: ["text"],
  sampled_rows: 64,
  sampled_bytes: 20480,
  warnings: [],
};

const IDLE_STATS: StreamStats = {
  state: "idle",
  rows_seen: 64,
  bytes_seen: 20480,
  batches_seen: 1,
  malformed_rows: 0,
  checksum: "ab12cd",
  progress: null,
};

beforeEach(() => {
  useEventStore.getState().reset();
  useUiStore.setState({ selectedDatasetId: null, toasts: [] });
});

describe("Datasets page", () => {
  function mockDatasetRoutes(info: DatasetInfo, stats: StreamStats = IDLE_STATS) {
    return mockFetch({
      "GET /api/datasets": { body: [info] },
      [`GET /api/datasets/${info.dataset_id}`]: { body: info },
      [`GET /api/datasets/${info.dataset_id}/stats`]: { body: stats },
      [`GET /api/datasets/${info.dataset_id}/preview`]: {
        body: { columns: ["text", "meta"], rows: [{ text: "hello world", meta: { a: 1 } }], truncated_values: false },
      },
      "POST /api/datasets": { body: info },
    });
  }

  it("shows unknown cardinality as 'Unknown — streaming' and never asks for a row count", async () => {
    mockDatasetRoutes(STREAMING_DS);
    useUiStore.getState().selectDataset("ds-1");
    renderWithProviders(<DatasetsPage />);

    expect(await screen.findByTestId("cardinality")).toHaveTextContent("Unknown — streaming");
    // No input anywhere asks for a number of rows.
    const inputs = screen.queryAllByRole("textbox").concat(screen.queryAllByRole("spinbutton"));
    for (const el of inputs) {
      const label = (el.getAttribute("aria-label") ?? "") + " " + (el.id ? document.querySelector(`label[for="${el.id}"]`)?.textContent ?? "" : "");
      expect(label.toLowerCase()).not.toMatch(/row/);
    }
    expect(screen.queryByLabelText(/rows?|row count/i)).toBeNull();
    expect(screen.queryAllByRole("spinbutton")).toHaveLength(0);

    // Columns, sample preview labelled "sample", stats without progress bar.
    expect(await screen.findByRole("table", { name: "Dataset columns" })).toBeInTheDocument();
    const preview = await screen.findByRole("table", { name: "Dataset sample preview" });
    expect(within(preview).getByText("hello world")).toBeInTheDocument();
    expect(screen.getAllByText(/sample/i).length).toBeGreaterThan(0);
    expect(await screen.findByText("ab12cd")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar", { name: "Scan progress" })).toBeNull();
    expect(screen.getByRole("button", { name: /full scan/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeDisabled();
  });

  it("shows a progress bar only when progress is known and enables Cancel while streaming", async () => {
    mockDatasetRoutes({ ...STREAMING_DS, cardinality: { rows: 1200, exact: true, method: "metadata" } }, { ...IDLE_STATS, state: "streaming", progress: 0.4 });
    useUiStore.getState().selectDataset("ds-1");
    renderWithProviders(<DatasetsPage />);
    expect(await screen.findByTestId("cardinality")).toHaveTextContent("1,200 rows");
    expect(await screen.findByRole("progressbar", { name: "Scan progress" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /full scan/i })).toBeDisabled();
  });

  it("loads a Hugging Face dataset via POST /api/datasets with the DatasetSpec", async () => {
    const f = mockDatasetRoutes(STREAMING_DS);
    renderWithProviders(<DatasetsPage />);
    fireEvent.change(screen.getByLabelText("Hugging Face dataset id"), { target: { value: "org/huge-corpus" } });
    fireEvent.click(screen.getByRole("button", { name: /load & inspect/i }));
    expect(await screen.findByTestId("cardinality")).toHaveTextContent("Unknown");
    const post = f.mock.calls.find(([u, i]) => String(u) === "/api/datasets" && i?.method === "POST");
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ source: "huggingface", name: "org/huge-corpus", split: "train", config_name: null });
    expect(useUiStore.getState().selectedDatasetId).toBe("ds-1");
  });
});

describe("Monitor page", () => {
  const status: TrainingStatus = {
    state: "running",
    job_id: "job-1",
    backend: "hf",
    step: 12,
    max_steps: 100,
    epoch: 0.12,
    loss: 1.2345,
    learning_rate: 0.0002,
    started_at: null,
    eta_seconds: null,
    supports_pause: false,
    message: "",
  };

  it("renders ETA as an em dash when null and disables Pause when unsupported", async () => {
    mockFetch({
      "GET /api/training/status": { body: status },
      "GET /api/training/metrics": { body: [{ step: 1, loss: 2 }, { step: 2, loss: 1.9 }] },
      "GET /api/resources": { body: { cpu_pct: 10, ram_used_bytes: 1, ram_total_bytes: 4 } },
      "GET /api/hardware": { status: 404 },
    });
    renderWithProviders(<MonitorPage />);
    await waitFor(() => expect(screen.getByTestId("step")).toHaveTextContent("12 / 100"));
    expect(screen.getByTestId("eta")).toHaveTextContent("—");
    expect(screen.getByText("1.2345")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /pause/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /stop/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /resume/i })).toBeDisabled();
    await waitFor(() => expect(useEventStore.getState().metrics).toHaveLength(2));
  });

  it("updates live from WS-fed store events (ETA, logs)", async () => {
    mockFetch({
      "GET /api/training/status": { body: { ...status, supports_pause: true } },
      "GET /api/training/metrics": { body: [] },
      "GET /api/resources": { body: { cpu_pct: 10, ram_used_bytes: 1, ram_total_bytes: 4 } },
      "GET /api/hardware": { status: 404 },
    });
    renderWithProviders(<MonitorPage />);
    await waitFor(() => expect(screen.getByTestId("step")).toHaveTextContent("12 / 100"));
    expect(screen.getByTestId("eta")).toHaveTextContent("—");
    act(() => {
      useEventStore.getState().apply({ seq: 1, type: "status", data: { ...status, supports_pause: true, eta_seconds: 125 } });
      useEventStore.getState().apply({ seq: 2, type: "log", data: { level: "info", message: "step 13 done" } });
    });
    expect(screen.getByTestId("eta")).toHaveTextContent("2m 05s");
    expect(screen.getByRole("log")).toHaveTextContent("step 13 done");
    expect(screen.getByRole("button", { name: /pause/i })).toBeEnabled();
  });
});

describe("Training page", () => {
  it("shows the max_steps-required hint when the selected dataset size is unknown", async () => {
    mockFetch({
      "GET /api/training/config": { body: { method: "qlora", learning_rate: 0.0002, num_epochs: 1, max_steps: null } },
      "GET /api/datasets": { body: [STREAMING_DS] },
      "GET /api/model": { body: { model_name: "unsloth/tiny", load_in_4bit: true } },
      "GET /api/training/status": { body: { state: "idle" } },
    });
    useUiStore.getState().selectDataset("ds-1");
    renderWithProviders(<TrainingPage />);
    expect(await screen.findByText("required: dataset size unknown")).toBeInTheDocument();
    // Starting without max_steps is blocked client-side with an inline error.
    fireEvent.click(screen.getByRole("button", { name: /start training/i }));
    expect(await screen.findByText("Required: dataset size unknown")).toBeInTheDocument();
  });

  it("displays server validation errors inline next to the field", async () => {
    mockFetch({
      "GET /api/training/config": { body: { method: "qlora", learning_rate: 0.0002, num_epochs: 1, max_steps: 10 } },
      "GET /api/datasets": { body: [{ ...STREAMING_DS, cardinality: { rows: 50, exact: true, method: "metadata" } }] },
      "GET /api/model": { body: { model_name: "unsloth/tiny" } },
      "GET /api/training/status": { body: { state: "idle" } },
      "PUT /api/training/config": {
        status: 422,
        body: { code: "validation_error", message: "invalid", detail: { errors: [{ loc: ["body", "lora_r"], msg: "Input should be less than or equal to 1024" }] } },
      },
    });
    useUiStore.getState().selectDataset("ds-1");
    renderWithProviders(<TrainingPage />);
    await screen.findByText("50 rows");
    fireEvent.click(screen.getByRole("button", { name: /save config/i }));
    const err = await screen.findByText("Input should be less than or equal to 1024");
    expect(screen.getByLabelText("LoRA r")).toHaveAttribute("aria-invalid", "true");
    expect(err).toHaveAttribute("role", "alert");
  });
});
