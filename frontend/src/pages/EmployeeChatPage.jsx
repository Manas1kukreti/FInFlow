import React from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import {
  FiAlertCircle,
  FiArrowRight,
  FiDatabase,
  FiDownload,
  FiMessageSquare,
  FiPlus,
  FiRefreshCw,
  FiSend,
  FiShield,
  FiSlash,
  FiStar,
  FiZap,
} from "react-icons/fi";
import PageHero from "../components/PageHero.jsx";
import StatCard from "../components/StatCard.jsx";
import StatusPill from "../components/StatusPill.jsx";
import DataTable from "../components/DataTable.jsx";
import VisualizationRenderer from "../components/visualization/VisualizationRenderer.jsx";
import { useWebSocket } from "../hooks/useWebSocket.js";
import {
  cancelChatExecution,
  createChatConversation,
  downloadChatExecutionResult,
  fetchChatConversation,
  fetchChatConversations,
  fetchChatDatasets,
  sendChatMessage,
} from "../api/finflow.js";

const ACTIVE_EXECUTION_STATUSES = new Set([
  "received",
  "interpreting",
  "grounding",
  "queued",
  "executing",
  "authorization_check",
  "composing_response",
]);

const RESULT_TYPE_LABELS = {
  scalar: "Scalar result",
  table: "Table result",
  chart: "Chart result",
  file: "File result",
  message: "Message",
};
const CHAT_CONVERSATION_STORAGE_KEY = "finflow.employeeChatConversationId";

export default function EmployeeChatPage() {
  const queryClient = useQueryClient();
  const scrollRef = useRef(null);
  const autoCreateRef = useRef(false);
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get("jobId") || "";
  const hasJobScope = Boolean(jobId);
  const previousJobIdRef = useRef(jobId);
  const [selectedConversationId, setSelectedConversationId] = useState(() =>
    jobId ? "" : (window.localStorage.getItem(CHAT_CONVERSATION_STORAGE_KEY) || ""),
  );
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [draft, setDraft] = useState("");
  const [queuedFileDownload, setQueuedFileDownload] = useState(false);

  const datasetsQuery = useQuery({
    queryKey: ["chat-datasets"],
    queryFn: fetchChatDatasets,
    enabled: hasJobScope,
  });

  const conversationsQuery = useQuery({
    queryKey: ["chat-conversations"],
    queryFn: fetchChatConversations,
    enabled: hasJobScope,
    refetchInterval: 5000,
  });

  useEffect(() => {
    if (previousJobIdRef.current === jobId) return;
    previousJobIdRef.current = jobId;
    setSelectedConversationId("");
    setSelectedDatasetId("");
    autoCreateRef.current = false;
  }, [jobId]);

  const selectedJobDataset = useMemo(
    () =>
      datasetsQuery.data?.find(
        (dataset) => String(dataset.source_submission_id || "") === String(jobId || ""),
      ) || null,
    [datasetsQuery.data, jobId],
  );
  const selectedJobConversation = useMemo(
    () =>
      selectedJobDataset
        ? conversationsQuery.data?.find(
            (conversation) =>
              String(conversation.active_dataset_id || "") === String(selectedJobDataset.dataset_id || ""),
          ) || null
        : null,
    [conversationsQuery.data, selectedJobDataset],
  );

  const createConversationMutation = useMutation({
    mutationFn: createChatConversation,
    onSuccess: async (conversation) => {
      setSelectedConversationId(conversation.id);
      if (!jobId) {
        window.localStorage.setItem(CHAT_CONVERSATION_STORAGE_KEY, conversation.id);
      }
      await queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
      await queryClient.invalidateQueries({ queryKey: ["chat-conversation", conversation.id] });
    },
    onError: () => {
      autoCreateRef.current = false;
    },
  });

  const conversationQuery = useQuery({
    queryKey: ["chat-conversation", selectedConversationId],
    queryFn: () => fetchChatConversation(selectedConversationId),
    enabled: hasJobScope && Boolean(selectedConversationId),
    retry: (failureCount, error) => {
      if (error?.response?.status === 404) return false;
      return failureCount < 2;
    },
    refetchInterval: (query) => {
      const executions = query.state.data?.executions || [];
      const active = executions.some((execution) => ACTIVE_EXECUTION_STATUSES.has(String(execution.status)));
      return active ? 2000 : false;
    },
  });
  const recoverableConversationNotFound =
    Boolean(selectedConversationId) && conversationQuery.error?.response?.status === 404;

  const sendMessageMutation = useMutation({
    mutationFn: ({ conversationId, message, datasetId }) =>
      sendChatMessage(conversationId, {
        message,
        client_message_id: crypto.randomUUID(),
        reply_to_message_id: null,
        dataset_id: datasetId || undefined,
      }),
    onSuccess: async (_response, variables) => {
      setDraft("");
      await queryClient.invalidateQueries({ queryKey: ["chat-conversation", variables.conversationId] });
      await queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
    },
  });

  const cancelExecutionMutation = useMutation({
    mutationFn: cancelChatExecution,
    onSuccess: async (_response, executionId) => {
      await queryClient.invalidateQueries({ queryKey: ["chat-conversation", selectedConversationId] });
      await queryClient.invalidateQueries({ queryKey: ["chat-execution", executionId] });
    },
  });

  const currentConversation = conversationQuery.data || null;
  const currentMessages = currentConversation?.messages || [];
  const currentExecutions = currentConversation?.executions || [];
  const activeDataset = useMemo(
    () => {
      const desiredDatasetId =
        currentConversation?.active_dataset_id || selectedDatasetId || selectedJobDataset?.dataset_id || "";
      const matchedDataset = datasetsQuery.data?.find(
        (dataset) => String(dataset.dataset_id) === String(desiredDatasetId),
      ) || null;
      if (matchedDataset) return matchedDataset;
      if (jobId) return selectedJobDataset || null;
      return datasetsQuery.data?.[0] || null;
    },
    [
      datasetsQuery.data,
      currentConversation?.active_dataset_id,
      jobId,
      selectedDatasetId,
      selectedJobDataset?.dataset_id,
    ],
  );

  useEffect(() => {
    if (selectedConversationId && !jobId) {
      window.localStorage.setItem(CHAT_CONVERSATION_STORAGE_KEY, selectedConversationId);
    }
  }, [jobId, selectedConversationId]);

  useEffect(() => {
    if (jobId) return;
    if (selectedConversationId || !conversationsQuery.data) return;
    const firstConversation = conversationsQuery.data[0];
    if (firstConversation) {
      setSelectedConversationId(firstConversation.id);
      window.localStorage.setItem(CHAT_CONVERSATION_STORAGE_KEY, firstConversation.id);
    }
  }, [conversationsQuery.data, jobId, selectedConversationId]);

  useEffect(() => {
    if (jobId || !selectedConversationId || !conversationsQuery.data) return;
    const selectedConversationExists = conversationsQuery.data.some(
      (conversation) => String(conversation.id) === String(selectedConversationId),
    );
    if (selectedConversationExists) return;
    window.localStorage.removeItem(CHAT_CONVERSATION_STORAGE_KEY);
    setSelectedConversationId("");
  }, [conversationsQuery.data, jobId, selectedConversationId]);

  useEffect(() => {
    if (!recoverableConversationNotFound) return;
    if (!jobId) {
      window.localStorage.removeItem(CHAT_CONVERSATION_STORAGE_KEY);
    }
    setSelectedConversationId("");
  }, [jobId, recoverableConversationNotFound]);

  useEffect(() => {
    if (selectedJobDataset?.dataset_id) {
      if (String(selectedDatasetId || "") !== String(selectedJobDataset.dataset_id || "")) {
        setSelectedDatasetId(selectedJobDataset.dataset_id);
      }
      return;
    }
    if (selectedDatasetId || !datasetsQuery.data?.length || jobId) return;
    setSelectedDatasetId(datasetsQuery.data[0].dataset_id);
  }, [datasetsQuery.data, jobId, selectedDatasetId, selectedJobDataset?.dataset_id]);

  useEffect(() => {
    if (autoCreateRef.current) return;
    if (conversationsQuery.isLoading || datasetsQuery.isLoading) return;
    if (selectedJobDataset?.dataset_id) {
      const jobDatasetId = String(selectedJobDataset.dataset_id || "");
      const currentConversationDatasetId = String(currentConversation?.active_dataset_id || "");
      const currentConversationMatchesJob = Boolean(jobDatasetId) && currentConversationDatasetId === jobDatasetId;
      if (currentConversationMatchesJob) return;
      if (selectedJobConversation?.id) {
        if (String(selectedConversationId || "") !== String(selectedJobConversation.id || "")) {
          setSelectedConversationId(selectedJobConversation.id);
        }
        return;
      }
      autoCreateRef.current = true;
      createConversationMutation.mutate({
        title: `Job chat: ${selectedJobDataset.display_name}`,
        dataset_id: selectedJobDataset.dataset_id,
      });
      return;
    }
    if (jobId) return;
    if (!conversationsQuery.data?.length && datasetsQuery.data?.length) {
      autoCreateRef.current = true;
      createConversationMutation.mutate({
        title: "Uploaded file chat",
        dataset_id: datasetsQuery.data[0].dataset_id,
      });
    }
  }, [
    conversationsQuery.data,
    conversationsQuery.isLoading,
    datasetsQuery.data,
    datasetsQuery.isLoading,
    selectedConversationId,
    selectedJobConversation?.id,
    selectedJobDataset?.dataset_id,
    createConversationMutation,
  ]);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [currentMessages.length, currentExecutions.length]);

  const handleWebSocketMessage = useCallback(
    (message) => {
      const conversationId = message?.payload?.conversation_id;
      if (conversationId) {
        queryClient.invalidateQueries({ queryKey: ["chat-conversation", conversationId] });
      }
      queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
    },
    [queryClient],
  );

  useWebSocket("employee-chat", handleWebSocketMessage);

  const activeExecution = useMemo(() => {
    if (!currentExecutions.length) return null;
    return [...currentExecutions].sort((left, right) => new Date(right.updated_at || right.created_at || 0) - new Date(left.updated_at || left.created_at || 0))[0];
  }, [currentExecutions]);

  const activeExecutionCount = currentExecutions.filter((execution) =>
    ACTIVE_EXECUTION_STATUSES.has(String(execution.status)),
  ).length;
  const visibleDatasets = useMemo(
    () => (jobId ? (selectedJobDataset ? [selectedJobDataset] : []) : (datasetsQuery.data || [])),
    [datasetsQuery.data, jobId, selectedJobDataset],
  );
  const visibleConversations = useMemo(
    () => (
      jobId
        ? (conversationsQuery.data || []).filter(
            (conversation) =>
              String(conversation.active_dataset_id || "") === String(selectedJobDataset?.dataset_id || ""),
          )
        : (conversationsQuery.data || [])
    ),
    [conversationsQuery.data, jobId, selectedJobDataset?.dataset_id],
  );

  async function handleSubmit(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || !selectedConversationId || sendMessageMutation.isPending) {
      return;
    }
    sendMessageMutation.mutate({
      conversationId: selectedConversationId,
      message,
      datasetId: selectedDatasetId || activeDataset?.dataset_id,
    });
  }

  async function handleDownload(execution) {
    if (!execution?.id) return;
    setQueuedFileDownload(true);
    try {
      await downloadChatExecutionResult(execution.id, execution?.result?.file?.format || "csv");
    } finally {
      setQueuedFileDownload(false);
    }
  }

  function startNewConversation() {
    const datasetId = selectedJobDataset?.dataset_id || selectedDatasetId || (!jobId ? datasetsQuery.data?.[0]?.dataset_id : "");
    if (!datasetId) return;
    createConversationMutation.mutate({
      title: selectedJobDataset?.display_name ? `Job chat: ${selectedJobDataset.display_name}` : "Uploaded file chat",
      dataset_id: datasetId,
    });
  }

  const conversations = visibleConversations;
  const datasets = visibleDatasets;

  if (!jobId) {
    return (
      <div className="ff-page-grid ff-chat-page">
        <PageHero
          eyebrow="Job chat"
          title="Open chat from a specific job"
          description="This workspace is now job-scoped. Open a job detail page first, then launch chat for that uploaded dataset."
        />
        <section className="ff-panel ff-chat-empty">
          <FiMessageSquare size={24} />
          <strong>Employee Chat is tied to one uploaded job at a time.</strong>
          <Link to="/jobs" className="ff-secondary-button">
            Return to My Jobs
          </Link>
        </section>
      </div>
    );
  }

  if (
    conversationsQuery.isLoading ||
    datasetsQuery.isLoading ||
    (selectedConversationId && conversationQuery.isLoading && !recoverableConversationNotFound)
  ) {
    return (
      <div className="ff-page-grid ff-chat-page">
        <PageHero
          eyebrow="Job chat"
          title="Loading your uploaded-file workspace"
          description="Preparing the chat catalog, conversation history, and execution stream."
        />
        <section className="ff-panel ff-chat-loading">
          <h3>Connecting to chat services...</h3>
        </section>
      </div>
    );
  }

  if (
    conversationsQuery.isError ||
    datasetsQuery.isError ||
    (selectedConversationId && conversationQuery.isError && !recoverableConversationNotFound)
  ) {
    return (
      <div className="ff-page-grid ff-chat-page">
        <PageHero
          eyebrow="Job chat"
          title="The chat workspace could not load"
          description="Check the backend is running and that the chat migration has been applied."
        />
        <section className="ff-panel ff-chat-empty">
          <FiAlertCircle size={24} />
          <strong>Chat data is unavailable right now.</strong>
        </section>
      </div>
    );
  }

  if (!selectedJobDataset) {
    return (
      <div className="ff-page-grid ff-chat-page">
        <PageHero
          eyebrow="Job chat"
          title="No chat dataset is attached to this job yet"
          description="The uploaded job could not be matched to a chat dataset. Re-open chat from the job detail page after the upload metadata finishes syncing."
        />
        <section className="ff-panel ff-chat-empty">
          <FiDatabase size={24} />
          <strong>This job is not ready for chat yet.</strong>
          <Link to={`/jobs/${encodeURIComponent(jobId)}`} className="ff-secondary-button">
            Back to job detail
          </Link>
        </section>
      </div>
    );
  }

  return (
    <div className="ff-page-grid ff-chat-page">
      <PageHero
        eyebrow="Job chat"
        title={`Ask FinFlow about ${selectedJobDataset.display_name}.`}
        description="This chat is scoped to the uploaded dataset for the current job. Ask for summaries, charts, filtered records, anomalies, and grounded row-level answers from this file only."
        actions={
          <>
            <button
              type="button"
              className="ff-primary-button"
              onClick={startNewConversation}
              disabled={createConversationMutation.isPending}
            >
              <FiPlus size={15} />
              New chat
            </button>
            <button
              type="button"
              className="ff-secondary-button"
              onClick={() => queryClient.invalidateQueries({ queryKey: ["chat-conversation", selectedConversationId] })}
            >
              <FiRefreshCw size={15} />
              Refresh
            </button>
          </>
        }
        aside={
          <div className="ff-chat-hero-aside">
            <div className="ff-chat-hero-aside__metric">
              <span>Scoped job</span>
              <strong>1</strong>
            </div>
            <div className="ff-chat-hero-aside__metric">
              <span>Running</span>
              <strong>{activeExecutionCount}</strong>
            </div>
            <div className="ff-chat-hero-aside__metric">
              <span>Active job</span>
              <strong>{activeDataset?.display_name || selectedJobDataset?.display_name || "No job selected"}</strong>
            </div>
          </div>
        }
      />

      <section className="ff-chat-summary-grid">
        <StatCard
          label="Conversation count"
          value={conversations.length}
          icon={<FiMessageSquare size={16} />}
          tone="complete"
        />
        <StatCard
          label="Live executions"
          value={activeExecutionCount}
          icon={<FiZap size={16} />}
          tone="running"
        />
        <StatCard
          label="Authorized columns"
          value={activeDataset?.columns?.length || 0}
          icon={<FiShield size={16} />}
          tone="complete"
        />
        <StatCard
          label="Current queue"
          value={activeExecution?.status || "idle"}
          icon={<FiDatabase size={16} />}
          tone={activeExecutionCount > 0 ? "running" : "complete"}
        />
      </section>

      <section className="ff-chat-grid">
        <aside className="ff-panel ff-chat-sidebar">
          <div className="ff-panel__head">
            <div>
              <p className="ff-eyebrow">Conversations</p>
              <h3>Recent threads</h3>
            </div>
          </div>

          <div className="ff-chat-conversation-list">
            {conversations.map((conversation) => {
              const isActive = String(conversation.id) === String(selectedConversationId);
              return (
                <button
                  key={conversation.id}
                  type="button"
                  className={`ff-chat-thread${isActive ? " is-active" : ""}`}
                  onClick={() => setSelectedConversationId(conversation.id)}
                >
                  <div className="ff-chat-thread__top">
                    <strong>{conversation.title}</strong>
                    <StatusPill status={conversation.status} />
                  </div>
                  <span>{conversation.last_message_preview || "No messages yet"}</span>
                  <small>
                    {conversation.active_dataset_id ? "Job attached" : "Job not selected"}
                  </small>
                </button>
              );
            })}
            {!conversations.length && (
              <div className="ff-chat-empty-state">
                <FiStar size={24} />
                <strong>No conversations yet</strong>
                <p>Start a new thread for this uploaded job.</p>
              </div>
            )}
          </div>

          <div className="ff-chat-sidebar__dataset">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Uploaded job</p>
                <h3>Scoped dataset</h3>
              </div>
            </div>
            <div className="ff-chat-conversation-list">
              {datasets.map((dataset) => {
                const isActive = String(dataset.dataset_id) === String(activeDataset?.dataset_id);
                return (
                  <div
                    key={dataset.dataset_id}
                    className={`ff-chat-thread${isActive ? " is-active" : ""}`}
                  >
                    <div className="ff-chat-thread__top">
                      <strong>{dataset.display_name}</strong>
                      <StatusPill status="active" />
                    </div>
                    <span>{dataset.description || "Uploaded file dataset"}</span>
                    <small>{dataset.source_submission_id ? `Job ${String(dataset.source_submission_id).slice(0, 8)}` : "Uploaded job"}</small>
                  </div>
                );
              })}
              {!datasets.length && (
                <div className="ff-chat-empty-state">
                  <FiDatabase size={24} />
                  <strong>No scoped dataset found</strong>
                  <p>Return to the job detail page and reopen chat for this upload.</p>
                </div>
              )}
            </div>
          </div>
        </aside>

        <main className="ff-panel ff-chat-main">
          <div className="ff-chat-main__top">
            <div>
              <p className="ff-eyebrow">Conversation</p>
              <h3>{currentConversation?.title || "New conversation"}</h3>
              <p className="ff-copy-muted">
                {currentConversation?.pending_clarification
                  ? currentConversation.pending_clarification.question
                  : "Ask about the uploaded file's rows, columns, summaries, charts, filters, or trends."}
              </p>
            </div>
            <div className="ff-chat-main__meta">
              <StatusPill status={activeExecution?.status || "idle"} />
              <button
                type="button"
                className="ff-ghost-button"
                onClick={() => cancelExecutionMutation.mutate(activeExecution?.id)}
                disabled={!activeExecution || !ACTIVE_EXECUTION_STATUSES.has(String(activeExecution.status)) || cancelExecutionMutation.isPending}
              >
                <FiSlash size={15} />
                Cancel
              </button>
            </div>
          </div>

          <div className="ff-chat-message-stream">
            {currentMessages.map((message) => (
              <ChatMessageCard
                key={message.id}
                message={message}
                onDownload={handleDownload}
                onCancel={cancelExecutionMutation.mutate}
              />
            ))}
            {!currentMessages.length && (
              <div className="ff-chat-welcome">
                <FiMessageSquare size={28} />
                <strong>Try a question like:</strong>
                <ul>
                  <li>Summarize the uploaded sheet in three bullets.</li>
                  <li>Show a bar chart for the top categories in this file.</li>
                  <li>Which rows look most unusual or incomplete?</li>
                </ul>
              </div>
            )}
            <div ref={scrollRef} />
          </div>

          <form className="ff-chat-composer" onSubmit={handleSubmit}>
            <label className="ff-chat-composer__field">
              <span>Ask a question</span>
              <textarea
                rows={3}
                placeholder="Ask about rows, columns, metrics, filters, or trends in the uploaded file..."
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
              />
            </label>
            <div className="ff-chat-composer__actions">
              <div className="ff-chat-composer__hint">
                <FiArrowRight size={14} />
                {selectedDatasetId || activeDataset?.dataset_id
                  ? "The conversation is grounded in the selected uploaded file."
                  : "Select an uploaded file before sending a question."}
              </div>
              <button
                type="submit"
                className="ff-primary-button"
                disabled={!draft.trim() || sendMessageMutation.isPending || !selectedConversationId}
              >
                <FiSend size={15} />
                {sendMessageMutation.isPending ? "Sending..." : "Send"}
              </button>
            </div>
          </form>
        </main>
      </section>

      {queuedFileDownload ? (
        <div className="ff-chat-download-banner">
          <FiDownload size={14} />
          Preparing file download...
        </div>
      ) : null}
    </div>
  );
}

function ChatMessageCard({ message, onDownload }) {
  const isUser = message.role === "user";
  const result = message.result || null;
  const clarification = message.clarification || null;
  const chartSpec = result?.chart || null;
  const isStatus = message.message_type === "status";
  const showPermission = message.message_type === "permission_denied";

  return (
    <article className={`ff-chat-message${isUser ? " is-user" : " is-assistant"}`}>
      <div className="ff-chat-message__topline">
        <StatusPill status={message.status || (isUser ? "user" : "assistant")} label={isUser ? "You" : RESULT_TYPE_LABELS[result?.result_type || message.message_type] || "FinFlow"} />
        <small>{new Date(message.created_at || Date.now()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
      </div>

      <div className="ff-chat-message__bubble">
        <p>{message.content}</p>
        {isStatus && !result ? (
          <div className="ff-chat-message__status">
            <FiRefreshCw className="is-spinning" />
            <span>Execution in progress</span>
          </div>
        ) : null}
        {clarification ? (
          <div className="ff-chat-clarification">
            <strong>{clarification.question}</strong>
            <div className="ff-chat-tags">
              {(clarification.options || []).map((option) => (
                <span key={option.id}>{option.label}</span>
              ))}
            </div>
          </div>
        ) : null}
        {result ? (
          <div className="ff-chat-result">
            {result.summary ? <p className="ff-copy-muted">{result.summary}</p> : null}
            {result.result_type === "scalar" ? (
              <div className="ff-chat-metric">
                <strong>{result.formatted_value || result.value || "—"}</strong>
                {result.unit ? <span>{result.unit}</span> : null}
              </div>
            ) : null}
            {chartSpec ? <VisualizationRenderer spec={chartSpec} /> : null}
            {Array.isArray(result.rows) && result.rows.length ? (
              <DataTable
                title={result.title || "Query result"}
                columns={(result.columns || []).map((column) => column.key)}
                rows={result.rows}
                pageSize={6}
              />
            ) : null}
            {result.file ? (
              <div className="ff-chat-file-card">
                <div>
                  <strong>{result.file.filename}</strong>
                  <span>{result.file.row_count} rows ready for download</span>
                </div>
                <button
                  type="button"
                  className="ff-secondary-button"
                  onClick={() => onDownload(message.execution_id ? { id: message.execution_id, result } : null)}
                  disabled={!message.execution_id}
                >
                  <FiDownload size={15} />
                  Download
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
        {showPermission ? (
          <div className="ff-chat-warning">
            <FiShield size={14} />
            <span>Authorization blocked this request. Try an aggregate summary instead.</span>
          </div>
        ) : null}
      </div>
    </article>
  );
}
