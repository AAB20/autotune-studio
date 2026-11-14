import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { SettingsPage } from "../pages/Settings";
import { CheckpointsPage } from "../pages/Checkpoints";
import { useUiStore } from "../services/uiStore";
import { mockFetch, renderWithProviders } from "./helpers";

const TOKEN = "hf_FAKEtestTOKEN1234567890";
const NOT_CONFIGURED = { configured: false, username: null, verified: false, can_write: null, token_role: null, message: "not configured" };
const CONNECTED = { configured: true, username: "alice", verified: true, can_write: true, token_role: "write", message: "verified" };

beforeEach(() => useUiStore.setState({ toasts: [] }));

describe("Settings page (Hugging Face)", () => {
  it("sends the token once, then clears it and never renders it", async () => {
    let sent: unknown = null;
    mockFetch({
      "GET /api/hf/credentials": { body: NOT_CONFIGURED },
      "PUT /api/hf/credentials": (init) => {
        sent = JSON.parse(String(init?.body));
        return { body: CONNECTED };
      },
    });
    const { container } = renderWithProviders(<SettingsPage />);
    expect(await screen.findByText("not connected")).toBeInTheDocument();

    const token = screen.getByLabelText("Access token") as HTMLInputElement;
    expect(token.type).toBe("password");
    fireEvent.change(screen.getByLabelText("Username or organisation"), { target: { value: "alice" } });
    fireEvent.change(token, { target: { value: TOKEN } });
    fireEvent.click(screen.getByRole("button", { name: /verify & save/i }));

    expect(await screen.findByText("connected as alice")).toBeInTheDocument();
    expect(sent).toEqual({ username: "alice", token: TOKEN });
    await waitFor(() => expect(token.value).toBe(""));
    expect(container.innerHTML).not.toContain(TOKEN);
  });
});

describe("Export panel push option", () => {
  it("is disabled until Hugging Face is connected", async () => {
    mockFetch({
      "GET /api/hf/credentials": { body: NOT_CONFIGURED },
      "GET /api/checkpoints": { body: [] },
      "GET /api/training/status": { body: { state: "idle", step: 0 } },
    });
    renderWithProviders(<CheckpointsPage />);
    expect(await screen.findByText(/connect your hugging face account in settings first/i)).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: /push to hugging face hub/i })).toBeDisabled();
  });

  it("sends push_to_hub with the repo name and shows the Hub link", async () => {
    let body: Record<string, unknown> = {};
    mockFetch({
      "GET /api/hf/credentials": { body: CONNECTED },
      "GET /api/checkpoints": { body: [] },
      "GET /api/training/status": { body: { state: "idle", step: 0 } },
      "POST /api/export": (init) => {
        body = JSON.parse(String(init?.body));
        return {
          body: {
            path: "/w/exports/x", format: "lora_adapter", files: ["adapter_model.safetensors"], size_bytes: 10,
            hub_repo_id: "alice/my-model", hub_url: "https://huggingface.co/alice/my-model",
          },
        };
      },
    });
    renderWithProviders(<CheckpointsPage />);
    const sw = await screen.findByRole("switch", { name: /push to hugging face hub/i });
    await waitFor(() => expect(sw).not.toBeDisabled());
    fireEvent.click(sw);
    fireEvent.change(await screen.findByLabelText("Repository name"), { target: { value: "my-model" } });
    fireEvent.click(screen.getByRole("button", { name: /export & push/i }));
    const link = await screen.findByRole("link", { name: /alice\/my-model/i });
    expect(link).toHaveAttribute("href", "https://huggingface.co/alice/my-model");
    expect(body).toMatchObject({ push_to_hub: true, hub_repo_name: "my-model", hub_private: true });
  });
});
