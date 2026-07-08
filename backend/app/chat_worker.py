from __future__ import annotations

import os
from uuid import UUID

import httpx
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models.chat import ChatConversation, ChatDataset, ChatExecution
from app.services.chat_runtime import build_chat_result
from app.services.json_safety import make_json_safe


async def _post_status(execution_id: UUID, payload: dict) -> None:
    backend_base_url = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    settings = get_settings()
    headers = {"x-agent-service-secret": settings.agent_service_secret}
    async with httpx.AsyncClient(base_url=backend_base_url, timeout=120.0) as client:
        await client.post(f"/api/employee-chat/internal/executions/{execution_id}/status", json=payload, headers=headers)


async def process_chat_execution(ctx, payload: dict) -> None:
    execution_id = UUID(str(payload.get("execution_id")))
    async with AsyncSessionLocal() as db:
        execution = (
            await db.execute(
                select(ChatExecution)
                .options(
                    selectinload(ChatExecution.conversation),
                    selectinload(ChatExecution.message),
                    selectinload(ChatExecution.dataset),
                )
                .where(ChatExecution.id == execution_id)
            )
        ).scalars().first()
        if execution is None:
            return
        if execution.status.value == "cancelled":
            return

        await _post_status(
            execution.id,
            {
                "status": "executing",
                "phase": "executing",
                "message_content": "Running the employee data query.",
                "message_type": "status",
            },
        )

        try:
            dataset = execution.dataset
            if dataset is None and execution.conversation and execution.conversation.active_dataset_id is not None:
                dataset = await db.get(ChatDataset, execution.conversation.active_dataset_id)
            if dataset is None:
                raise RuntimeError("No dataset is available for this execution.")

            frame = await _load_dataset_frame(db, dataset.id)
            if frame.empty:
                raise RuntimeError("The selected dataset does not contain any rows.")

            intent_payload = execution.grounded_intent_json if isinstance(execution.grounded_intent_json, dict) else execution.canonical_intent_json
            if not isinstance(intent_payload, dict):
                raise RuntimeError("Execution is missing a canonical intent payload.")

            result = build_chat_result(
                execution_id=execution.id,
                dataset=dataset,
                intent=make_json_safe(intent_payload),
                frame=frame,
            )
            await _post_status(
                execution.id,
                {
                    "status": "completed",
                    "phase": "completed",
                    "result": result,
                    "message_type": result.get("result_type", "result"),
                    "message_content": result.get("summary") or "Employee chat execution completed.",
                },
            )
        except Exception as exc:
            await _post_status(
                execution.id,
                {
                    "status": "failed",
                    "phase": "completed",
                    "error_code": "EXECUTION_FAILED",
                    "error_message": str(exc),
                    "message_type": "error",
                    "message_content": str(exc) or "Employee chat execution failed.",
                },
            )


async def _load_dataset_frame(db, dataset_id: UUID):
    from app.services.chat_catalog import load_dataset_frame

    return await load_dataset_frame(db, dataset_id)


async def worker_startup(ctx):
    ctx["backend_base_url"] = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000")


class WorkerSettings:
    functions = [process_chat_execution]
    on_startup = worker_startup
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    queue_name = get_settings().chat_dispatch_queue
