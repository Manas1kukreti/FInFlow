from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatDatasetColumnRead(BaseModel):
    physical_column: str
    semantic_name: str | None = None
    description: str | None = None
    data_type: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    null_percentage: float | None = None
    distinct_count: int | None = None
    synonyms: list[str] = Field(default_factory=list)
    sensitivity: str | None = None


class ChatDatasetRead(BaseModel):
    dataset_id: UUID
    dataset_key: str
    source_submission_id: UUID | None = None
    display_name: str
    description: str
    domain: str
    storage_location: str
    schema_version: str
    aliases: list[str] = Field(default_factory=list)
    columns: list[ChatDatasetColumnRead] = Field(default_factory=list)
    sample_values: dict = Field(default_factory=dict)
    owner: str = ""
    sensitivity: str = "internal"
    allowed_roles: list[str] = Field(default_factory=list)
    chat_enabled: bool = True
    last_updated_at: datetime | None = None


class ChatConversationCreateRequest(BaseModel):
    title: str | None = None
    dataset_id: UUID | None = None


class ChatConversationSummaryRead(BaseModel):
    id: UUID
    title: str
    status: str
    employee_id: UUID
    active_dataset_id: UUID | None = None
    last_message_preview: str | None = None
    last_message_at: datetime | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None


class ChatClarificationOptionRead(BaseModel):
    id: str
    label: str
    description: str | None = None


class ChatClarificationRead(BaseModel):
    question: str
    options: list[ChatClarificationOptionRead] = Field(default_factory=list)
    allow_free_text: bool = True
    pending_intent_revision_id: UUID | None = None
    reason_code: str | None = None


class ChatResultColumnRead(BaseModel):
    key: str
    label: str
    data_type: str
    format: str | None = None


class ChatChartAxisRead(BaseModel):
    field: str
    label: str


class ChatChartSeriesRead(BaseModel):
    field: str
    label: str
    data_type: str | None = None


class ChatChartRead(BaseModel):
    chart_type: str
    title: str
    x_axis: ChatChartAxisRead | None = None
    y_axis: ChatChartAxisRead | None = None
    series: list[ChatChartSeriesRead] = Field(default_factory=list)
    data: list[dict] = Field(default_factory=list)
    status: str = "ready"
    encoding: dict = Field(default_factory=dict)
    options: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ChatResultRead(BaseModel):
    result_type: str
    title: str | None = None
    summary: str | None = None
    value: object | None = None
    formatted_value: str | None = None
    unit: str | None = None
    columns: list[ChatResultColumnRead] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    chart: ChatChartRead | None = None
    file: dict | None = None


class ChatMessageRead(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    message_type: str
    content: str
    status: str
    canonical_intent: dict | None = None
    grounded_intent: dict | None = None
    result: ChatResultRead | None = None
    clarification: ChatClarificationRead | None = None
    execution_id: UUID | None = None
    reply_to_message_id: UUID | None = None
    error_code: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatExecutionRead(BaseModel):
    id: UUID
    conversation_id: UUID
    message_id: UUID
    dataset_id: UUID | None = None
    status: str
    phase: str
    canonical_intent: dict | None = None
    grounded_intent: dict | None = None
    result: ChatResultRead | None = None
    error_code: str | None = None
    error_message: str | None = None
    client_message_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    updated_at: datetime | None = None


class ChatConversationDetailRead(ChatConversationSummaryRead):
    context: dict = Field(default_factory=dict)
    pending_clarification: ChatClarificationRead | None = None
    last_successful_intent: dict | None = None
    last_result: ChatResultRead | None = None
    messages: list[ChatMessageRead] = Field(default_factory=list)
    executions: list[ChatExecutionRead] = Field(default_factory=list)


class ChatSendMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    reply_to_message_id: UUID | None = None
    client_message_id: str | None = Field(default=None, max_length=128)


class ChatSendMessageResponse(BaseModel):
    message_id: UUID
    execution_id: UUID | None = None
    status: str
    message_type: str | None = None
    conversation_id: UUID


class ChatCancelExecutionResponse(BaseModel):
    execution_id: UUID
    status: str


class ChatExportRequest(BaseModel):
    format: str = Field(default="csv", pattern="^(csv|json)$")


class ChatExecutionStatusRead(BaseModel):
    execution_id: UUID
    status: str
    phase: str
    conversation_id: UUID
    message_id: UUID
    result: ChatResultRead | None = None
    error_code: str | None = None
    error_message: str | None = None
    updated_at: datetime | None = None
