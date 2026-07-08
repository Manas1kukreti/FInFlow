import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import EmployeeChatPage from "./EmployeeChatPage.jsx";

vi.mock("../hooks/useWebSocket.js", () => ({
  useWebSocket: () => {},
}));

vi.mock("../api/finflow.js", () => ({
  fetchChatDatasets: vi.fn().mockResolvedValue([
    {
      dataset_id: "dataset-1",
      dataset_key: "job:submission-1",
      source_submission_id: "submission-1",
      display_name: "Payroll Extract",
      description: "Uploaded spreadsheet",
      domain: "uploads",
      columns: [
        { physical_column: "department", semantic_name: "department" },
        { physical_column: "salary", semantic_name: "salary" },
      ],
    },
  ]),
  fetchChatConversations: vi.fn().mockResolvedValue([
    {
      id: "conversation-1",
      title: "Uploaded file chat",
      status: "active",
      employee_id: "employee-1",
      active_dataset_id: "dataset-1",
      last_message_preview: "How many employees are there?",
      last_message_at: null,
      updated_at: null,
      created_at: null,
    },
  ]),
  fetchChatConversation: vi.fn().mockResolvedValue({
    id: "conversation-1",
    title: "Uploaded file chat",
    status: "active",
    employee_id: "employee-1",
    active_dataset_id: "dataset-1",
    last_message_preview: "How many employees are there?",
    last_message_at: null,
    updated_at: null,
    created_at: null,
    context: {},
    pending_clarification: null,
    last_successful_intent: null,
    last_result: null,
    messages: [],
    executions: [],
  }),
  createChatConversation: vi.fn().mockResolvedValue({ id: "conversation-1" }),
  sendChatMessage: vi.fn().mockResolvedValue({
    message_id: "message-1",
    execution_id: "execution-1",
    status: "queued",
    message_type: "status",
    conversation_id: "conversation-1",
  }),
  cancelChatExecution: vi.fn().mockResolvedValue({}),
  downloadChatExecutionResult: vi.fn().mockResolvedValue(undefined),
}));

describe("EmployeeChatPage", () => {
  it("renders the uploaded-file chat workspace", async () => {
    window.localStorage.clear();
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <MemoryRouter initialEntries={["/employee-chat?jobId=submission-1"]}>
        <QueryClientProvider client={queryClient}>
          <EmployeeChatPage />
        </QueryClientProvider>
      </MemoryRouter>,
    );

    await screen.findByRole("button", { name: /send/i });
    await screen.findByRole("heading", { name: "Scoped dataset" });
    expect(screen.getAllByText("Payroll Extract").length).toBeGreaterThan(0);
    expect(screen.getByText("Ask FinFlow about Payroll Extract.")).toBeInTheDocument();
  });

  it("requires a job-specific entry point", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <MemoryRouter initialEntries={["/employee-chat"]}>
        <QueryClientProvider client={queryClient}>
          <EmployeeChatPage />
        </QueryClientProvider>
      </MemoryRouter>,
    );

    await screen.findByText("Open chat from a specific job");
    expect(screen.getByRole("link", { name: /return to my jobs/i })).toHaveAttribute("href", "/jobs");
  });
});
