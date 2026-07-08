from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import String, and_, case, cast, desc, distinct, func, not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_roles
from app.db.session import get_db
from app.models import Review, Submission, SubmissionComment, SubmissionRecord, SubmissionStatus, User, UserRole, normalize_submission_status


router = APIRouter(prefix="/analytics", tags=["analytics"])


def _awaiting_agent_condition():
    payload_status = func.coalesce(Submission.summary["status"].astext, "")
    return and_(
        cast(Submission.status, String).in_(["pending", SubmissionStatus.queued.value, SubmissionStatus.quarantined.value]),
        payload_status.in_(["pending_agent_availability", "rejected"]),
    )


def _canonical_submission_status_expr():
    status = cast(Submission.status, String)
    payload_status = func.coalesce(Submission.summary["status"].astext, "")
    return case(
        (_awaiting_agent_condition(), SubmissionStatus.quarantined.value),
        (status == "pending", SubmissionStatus.queued.value),
        (status == "processing", SubmissionStatus.running.value),
        (status == "complete", SubmissionStatus.succeeded.value),
        (status == "success", SubmissionStatus.succeeded.value),
        (status == "partial", SubmissionStatus.failed.value),
        (status == "rejected", SubmissionStatus.quarantined.value),
        else_=status,
    )


def _derived_submission_status_expr():
    return _canonical_submission_status_expr()


def _kpi_trend_base(role_filters: list, date_filters: list):
    return (
        select(
            func.date(Submission.uploaded_at).label("day"),
            Submission.id.label("submission_id"),
            _derived_submission_status_expr().label("status"),
        )
        .select_from(Submission)
        .where(*role_filters, *date_filters)
        .subquery("kpi_trend_base")
    )


def _kpi_trend_rows_query(kpi_trend_base):
    return (
        select(
            kpi_trend_base.c.day,
            kpi_trend_base.c.status,
            func.count(distinct(kpi_trend_base.c.submission_id)).label("uploads"),
            func.count(SubmissionRecord.id).label("records"),
        )
        .select_from(kpi_trend_base)
        .join(SubmissionRecord, SubmissionRecord.submission_id == kpi_trend_base.c.submission_id, isouter=True)
        .group_by(kpi_trend_base.c.day, kpi_trend_base.c.status)
        .order_by(kpi_trend_base.c.day)
        .limit(90)
    )


def _rollup_kpi_trend_rows(rows) -> list[dict]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        day_key = row.day.isoformat() if row.day else ""
        bucket = grouped.setdefault(
            day_key,
            {
                "uploads": 0,
                SubmissionStatus.queued.value: 0,
                SubmissionStatus.planning.value: 0,
                SubmissionStatus.running.value: 0,
                SubmissionStatus.succeeded.value: 0,
                SubmissionStatus.failed.value: 0,
                SubmissionStatus.quarantined.value: 0,
                SubmissionStatus.callback_failed.value: 0,
                SubmissionStatus.awaiting_schema_approval.value: 0,
                SubmissionStatus.awaiting_confirmation.value: 0,
                SubmissionStatus.declined.value: 0,
                "records": 0,
            },
        )
        bucket["uploads"] += int(row.uploads or 0)
        bucket["records"] += int(row.records or 0)
        bucket[str(row.status)] = bucket.get(str(row.status), 0) + int(row.uploads or 0)

    return [
        {
            "day": day or None,
            "uploads": values["uploads"],
            "queued": values[SubmissionStatus.queued.value],
            "planning": values[SubmissionStatus.planning.value],
            "running": values[SubmissionStatus.running.value],
            "succeeded": values[SubmissionStatus.succeeded.value],
            "failed": values[SubmissionStatus.failed.value] + values[SubmissionStatus.callback_failed.value],
            "quarantined": values[SubmissionStatus.quarantined.value],
            "callback_failed": values[SubmissionStatus.callback_failed.value],
            "awaiting_schema_approval": values[SubmissionStatus.awaiting_schema_approval.value],
            "awaiting_confirmation": values[SubmissionStatus.awaiting_confirmation.value],
            "declined": values[SubmissionStatus.declined.value],
            "complete": values[SubmissionStatus.succeeded.value],
            "awaiting_agent": values[SubmissionStatus.quarantined.value],
            "pending": values[SubmissionStatus.queued.value] + values[SubmissionStatus.planning.value],
            "processing": values[SubmissionStatus.running.value],
            "records": values["records"],
        }
        for day, values in sorted(grouped.items())
    ]


def _agent_summary_text(payload: dict, agent_id: str) -> str:
    agent_summaries = payload.get("agent_summaries")
    if isinstance(agent_summaries, list):
        for entry in agent_summaries:
            if not isinstance(entry, dict):
                continue
            current_agent_id = str(entry.get("agent_id", "")).strip().lower()
            if current_agent_id != agent_id:
                continue
            summary = " ".join(str(entry.get("summary", "")).split()).strip()
            if summary:
                return summary[:-1] if summary.endswith(".") else summary
            bullets = entry.get("bullets")
            if isinstance(bullets, list):
                for bullet in bullets:
                    text = " ".join(str(bullet).split()).strip()
                    if text:
                        return text[:-1] if text.endswith(".") else text
    return ""


@router.get("/kpis")
async def get_kpis(
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> dict:
    role_filters = _role_filters_no_status(user)
    date_filters = _date_filters(date_from, date_to)

    scoped_submission_ids = (
        select(Submission.id)
        .where(*role_filters, *date_filters)
        .distinct()
        .subquery("scoped_submission_ids")
    )
    scoped_submission_ids_select = select(scoped_submission_ids.c.id)

    derived_status = _derived_submission_status_expr().label("status")
    status_rows = (
        await db.execute(
            select(derived_status, func.count(Submission.id).label("cnt"))
            .where(Submission.id.in_(scoped_submission_ids_select))
            .group_by(derived_status)
        )
    ).all()
    status_counts: dict[str, int] = {str(row.status): row.cnt for row in status_rows}

    total_submissions = sum(status_counts.values())
    queued_count = status_counts.get(SubmissionStatus.queued.value, 0)
    planning_count = status_counts.get(SubmissionStatus.planning.value, 0)
    running_count = status_counts.get(SubmissionStatus.running.value, 0)
    succeeded_count = status_counts.get(SubmissionStatus.succeeded.value, 0)
    failed_count = status_counts.get(SubmissionStatus.failed.value, 0)
    quarantined_count = status_counts.get(SubmissionStatus.quarantined.value, 0)
    callback_failed_count = status_counts.get(SubmissionStatus.callback_failed.value, 0)
    awaiting_schema_count = status_counts.get(SubmissionStatus.awaiting_schema_approval.value, 0)
    awaiting_confirmation_count = status_counts.get(SubmissionStatus.awaiting_confirmation.value, 0)
    declined_count = status_counts.get(SubmissionStatus.declined.value, 0)
    completion_rate = succeeded_count / total_submissions if total_submissions else 0.0

    total_records: int = await db.scalar(
        select(func.count(SubmissionRecord.id))
        .join(Submission, SubmissionRecord.submission_id == Submission.id)
        .where(*role_filters, *date_filters)
    ) or 0

    average_review_seconds: float = await db.scalar(
        select(func.avg(func.extract("epoch", Submission.completed_at - Submission.uploaded_at)))
        .select_from(Submission)
        .where(
            Submission.id.in_(scoped_submission_ids_select),
            Submission.completed_at.is_not(None)
        )
    ) or 0.0

    output_rows = (
        await db.execute(
            select(Submission.output_format, func.count(Submission.id).label("cnt"))
            .where(Submission.id.in_(scoped_submission_ids_select))
            .group_by(Submission.output_format)
            .order_by(desc("cnt"))
        )
    ).all()
    output_breakdown = [
        {"output_format": row.output_format or "Unknown", "count": row.cnt}
        for row in output_rows
    ]

    trend_base = _kpi_trend_base(role_filters, date_filters)
    trend_rows = (await db.execute(_kpi_trend_rows_query(trend_base))).all()
    upload_trends = _rollup_kpi_trend_rows(trend_rows)

    recent_submissions = (
        await db.execute(
            select(Submission)
            .where(Submission.id.in_(scoped_submission_ids_select))
            .order_by(desc(Submission.uploaded_at))
            .limit(5)
        )
    ).scalars().all()

    latest_submission = await db.scalar(
        select(Submission)
        .where(*role_filters, *date_filters)
        .order_by(desc(Submission.uploaded_at))
        .limit(1)
    )

    latest_records: list[SubmissionRecord] = []
    if latest_submission:
        latest_records = (
            await db.execute(
                select(SubmissionRecord)
                .where(SubmissionRecord.submission_id == latest_submission.id)
                .order_by(SubmissionRecord.record_index)
                .limit(10)
            )
        ).scalars().all()

    daily_record_trends = upload_trends

    planner_rows = (
        await db.execute(
            select(
                Submission.id,
                Submission.file_name,
                Submission.uploaded_at,
                Submission.completed_at,
                Submission.summary,
            )
            .where(Submission.id.in_(scoped_submission_ids_select))
            .order_by(desc(Submission.completed_at), desc(Submission.uploaded_at))
        )
    ).all()

    planner_run_count = 0
    planner_runs: list[dict] = []
    latest_planned_at = None
    latest_plan_summary = ""
    for row in planner_rows:
        payload = row.summary if isinstance(row.summary, dict) else {}
        summary = _agent_summary_text(payload, "orchestrator")
        if not summary:
            continue

        planner_run_count += 1
        planned_at = row.completed_at or row.uploaded_at
        if planned_at and (latest_planned_at is None or planned_at > latest_planned_at):
            latest_planned_at = planned_at
            latest_plan_summary = summary

        if len(planner_runs) < 5:
            planner_runs.append(
                {
                    "upload_id": row.id,
                    "filename": row.file_name,
                    "summary": summary,
                    "planned_at": planned_at.isoformat() if planned_at else None,
                }
            )

    return {
        "totals": {
            "uploads": total_submissions,
            "queued": queued_count,
            "planning": planning_count,
            "running": running_count,
            "succeeded": succeeded_count,
            "failed": failed_count + callback_failed_count,
            "quarantined": quarantined_count,
            "callback_failed": callback_failed_count,
            "awaiting_schema_approval": awaiting_schema_count,
            "awaiting_confirmation": awaiting_confirmation_count,
            "declined": declined_count,
            "complete": succeeded_count,
            "pending": queued_count + planning_count,
            "processing": running_count,
            "awaiting_agent": quarantined_count,
            "approved": succeeded_count,
            "completion_rate": round(completion_rate * 100, 1),
            "average_review_seconds": float(average_review_seconds),
            "rows": total_records,
            "structured_records": total_records,
            "approved_amount": 0,
            "cash": 0,
        },
        "date_mode": "uploaded_at",
        "output_breakdown": output_breakdown,
        "account_class_breakdown": [],
        "daily_trends": [
            {
                "date": row["day"],
                "uploads": row["uploads"],
                "queued": row["queued"],
                "planning": row["planning"],
                "running": row["running"],
                "succeeded": row["succeeded"],
                "failed": row["failed"],
                "quarantined": row["quarantined"],
                "callback_failed": row["callback_failed"],
                "awaiting_schema_approval": row["awaiting_schema_approval"],
                "awaiting_confirmation": row["awaiting_confirmation"],
                "declined": row["declined"],
                "complete": row["succeeded"],
                "awaiting_agent": row["quarantined"],
                "pending": row["queued"] + row["planning"],
                "processing": row["running"],
                "row_count": row["records"],
            }
            for row in daily_record_trends
        ],
        "daily_transaction_trends": [
            {
                "date": row["day"],
                "total": row["records"],
                "completed": row["succeeded"],
                "under_investigation": row["failed"] + row["callback_failed"] + row["declined"],
            }
            for row in daily_record_trends
        ],
        "upload_trends": upload_trends,
        "recent_uploads": [
            {
                "id": submission.id,
                "filename": submission.file_name,
                "status": _serialize_submission_status(submission),
                "rows": await _record_count_for_submission(db, submission.id),
                "created_at": submission.uploaded_at,
            }
            for submission in recent_submissions
        ],
        "latest_upload": {
            "id": latest_submission.id,
            "filename": latest_submission.file_name,
            "status": _serialize_submission_status(latest_submission),
            "created_at": latest_submission.uploaded_at,
        } if latest_submission else None,
        "last_rows": [record.payload for record in latest_records],
        "last_transactions": [_format_latest_record(record, latest_submission) for record in latest_records],
        "transaction_amount_trend": [
            {"date": row["day"], "amount": row["records"]}
            for row in daily_record_trends
        ],
        "workflow_amounts": {
            "initiated": total_submissions,
            "pending": queued_count + planning_count,
            "awaiting_agent": quarantined_count,
            "approved": succeeded_count,
            "declined": declined_count,
        },
        "personal": {
            "scope": user.role.value,
            "average_review_seconds": float(average_review_seconds),
            "completion_rate": round(completion_rate * 100, 1),
            "failure_reasons": await _failure_reasons(db, scoped_submission_ids),
            "rejection_reasons": await _failure_reasons(db, scoped_submission_ids),
        },
        "workflow_planner": {
            "total_runs": planner_run_count,
            "last_run_at": latest_planned_at.isoformat() if latest_planned_at else None,
            "latest_summary": latest_plan_summary,
            "recent_runs": planner_runs,
        },
        "kpi_snapshots": [],
    }


def _date_filters(date_from: datetime | None, date_to: datetime | None) -> list:
    filters = []
    if date_from:
        filters.append(Submission.uploaded_at >= date_from)
    if date_to:
        filters.append(Submission.uploaded_at <= date_to)
    return filters


def _role_filters_no_status(user: User) -> list:
    if user.role == UserRole.employee:
        return [Submission.user_id == user.id]
    if user.role == UserRole.manager:
        return [Submission.user_id.in_(select(User.id).where(User.manager_id == user.id))]
    return []


async def _record_count_for_submission(db: AsyncSession, submission_id) -> int:
    return await db.scalar(
        select(func.count()).select_from(SubmissionRecord).where(SubmissionRecord.submission_id == submission_id)
    ) or 0


async def _failure_reasons(db: AsyncSession, scoped_submission_ids) -> list[dict]:
    scoped_submission_ids_select = select(scoped_submission_ids.c.id)
    rows = (
        await db.execute(
            select(Submission, SubmissionComment)
            .select_from(Submission)
            .join(SubmissionComment, SubmissionComment.submission_id == Submission.id)
            .where(
                Submission.id.in_(scoped_submission_ids_select),
                cast(Submission.status, String).in_([
                    SubmissionStatus.failed.value,
                    SubmissionStatus.callback_failed.value,
                    SubmissionStatus.declined.value,
                ]),
            )
            .order_by(desc(Submission.uploaded_at), desc(SubmissionComment.created_at))
            .limit(25)
        )
    ).all()

    reasons = []
    seen: set = set()
    for submission, comment in rows:
        if submission.id in seen:
            continue
        seen.add(submission.id)
        reasons.append(
            {
                "upload_id": submission.id,
                "filename": submission.file_name,
                "status": normalize_submission_status(submission.status),
                "reason": comment.message,
                "created_at": comment.created_at,
            }
        )
        if len(reasons) >= 5:
            break
    return reasons


async def _daily_record_trends(db: AsyncSession, role_filters: list, date_filters: list) -> list[dict]:
    rows = (await db.execute(_kpi_trend_rows_query(_kpi_trend_base(role_filters, date_filters)))).all()
    return _rollup_kpi_trend_rows(rows)


def _format_latest_record(record: SubmissionRecord, submission: Submission | None) -> dict:
    payload = record.payload if isinstance(record.payload, dict) else {}
    first_text = next((str(value) for value in payload.values() if isinstance(value, str) and value.strip()), "-")
    return {
        "upload_id": str(record.submission_id),
        "row_index": record.record_index,
        "transaction_id": f"Record {record.record_index}",
        "transaction_date": submission.uploaded_at.isoformat() if submission and submission.uploaded_at else None,
        "merchant_name": first_text,
        "transaction_type": submission.output_format if submission else "-",
        "payment_method": "-",
        "amount": 0,
        "status": _serialize_submission_status(submission) if submission else "complete",
    }


def _serialize_submission_status(submission: Submission | None) -> str:
    if submission is None:
        return SubmissionStatus.succeeded.value
    summary_payload = submission.summary if isinstance(submission.summary, dict) else {}
    payload_status = str(summary_payload.get("status", "")).strip().lower()
    status_str = normalize_submission_status(submission.status)
    if status_str == SubmissionStatus.queued.value and payload_status in {"pending_agent_availability", "rejected"}:
        return SubmissionStatus.quarantined.value
    return status_str
