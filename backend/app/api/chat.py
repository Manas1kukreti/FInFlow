from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import require_roles
from app.db.session import get_db
from app.models import User, UserRole
from app.schemas.chat import (
    ChatConversationCreateRequest,
    ChatConversationDetailRead,
    ChatConversationSummaryRead,
    ChatDatasetRead,
    ChatExecutionRead,
    ChatSendMessageRequest,
    ChatSendMessageResponse,
)
from app.services.chat_runtime import (
    CHAT_WORKFLOW_NAME,
    CHAT_WORKER_SECRET_HEADER,
    cancel_chat_execution,
    create_chat_conversation,
    download_chat_execution_result,
    finalize_execution_update,
    get_chat_conversation_detail,
    get_chat_execution,
    list_chat_conversations,
    list_chat_datasets_for_user,
    submit_chat_message,
)

router = APIRouter(prefix="/employee-chat", tags=["employee-chat"])


async def _get_chat_redis(request: Request):
    redis = getattr(request.app.state, "chat_dispatch_redis", None)
    if redis is not None:
        return redis, False
    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return redis, True


async def _enqueue_chat_job(request: Request, execution_id) -> None:
    settings = get_settings()
    redis, needs_close = await _get_chat_redis(request)
    try:
        await redis.enqueue_job(
            "process_chat_execution",
            {"execution_id": str(execution_id)},
            _job_id=f"chat:{execution_id}",
            _queue_name=settings.chat_dispatch_queue,
        )
    finally:
        if needs_close:
            await redis.close()


@router.get("/datasets", response_model=list[ChatDatasetRead])
async def get_datasets(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> list[ChatDatasetRead]:
    return await list_chat_datasets_for_user(db, user)


@router.get("/conversations", response_model=list[ChatConversationSummaryRead])
async def get_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> list[ChatConversationSummaryRead]:
    return await list_chat_conversations(db, user)


@router.post("/conversations", response_model=ChatConversationDetailRead, status_code=status.HTTP_201_CREATED)
async def post_conversation(
    payload: ChatConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ChatConversationDetailRead:
    return await create_chat_conversation(db, user=user, payload=payload)


@router.get("/conversations/{conversation_id}", response_model=ChatConversationDetailRead)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ChatConversationDetailRead:
    return await get_chat_conversation_detail(db, conversation_id, user=user)


@router.post("/conversations/{conversation_id}/messages", response_model=ChatSendMessageResponse)
async def post_message(
    conversation_id: UUID,
    payload: ChatSendMessageRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ChatSendMessageResponse:
    outcome = await submit_chat_message(db, user=user, conversation_id=conversation_id, payload=payload)
    if outcome.execution is not None and not outcome.denied and outcome.response.status == "queued":
        try:
            await _enqueue_chat_job(request, outcome.execution.id)
        except Exception as exc:
            await finalize_execution_update(
                db,
                execution_id=outcome.execution.id,
                status_value="failed",
                phase_value="completed",
                error_code="QUEUE_FAILED",
                error_message=str(exc),
                message_type="error",
                message_content="The employee chat queue is temporarily unavailable.",
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="The employee chat queue is temporarily unavailable.") from exc
    return outcome.response


@router.get("/executions/{execution_id}", response_model=ChatExecutionRead)
async def get_execution(
    execution_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ChatExecutionRead:
    return await get_chat_execution(db, execution_id=execution_id, user=user)


@router.post("/executions/{execution_id}/cancel", response_model=ChatExecutionRead)
async def cancel_execution(
    execution_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ChatExecutionRead:
    return await cancel_chat_execution(db, execution_id=execution_id, user=user)


@router.get("/executions/{execution_id}/download")
async def download_execution(
    execution_id: UUID,
    request: Request,
    format: str = "csv",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> Response:
    filename, media_type, body = await download_chat_execution_result(
        db,
        execution_id=execution_id,
        user=user,
        format_name=format,
    )
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=body, media_type=media_type, headers=headers)


@router.post("/internal/executions/{execution_id}/status", response_model=ChatExecutionRead)
async def update_execution_status_internal(
    execution_id: UUID,
    payload: dict,
    x_agent_service_secret: str | None = Header(default=None, alias=CHAT_WORKER_SECRET_HEADER),
    db: AsyncSession = Depends(get_db),
) -> ChatExecutionRead:
    settings = get_settings()
    if x_agent_service_secret != settings.agent_service_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid employee chat service secret")
    return await finalize_execution_update(
        db,
        execution_id=execution_id,
        status_value=str(payload.get("status", "failed")),
        phase_value=str(payload.get("phase")) if payload.get("phase") else None,
        result_json=payload.get("result") if isinstance(payload.get("result"), dict) else None,
        error_code=str(payload.get("error_code")) if payload.get("error_code") else None,
        error_message=str(payload.get("error_message")) if payload.get("error_message") else None,
        message_type=str(payload.get("message_type")) if payload.get("message_type") else None,
        message_content=str(payload.get("message_content")) if payload.get("message_content") else None,
    )
