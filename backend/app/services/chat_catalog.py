from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Submission, User, UserRole
from app.models.chat import ChatDataset, ChatDatasetRow
from app.services.data_profile import build_data_profile_from_file
from app.services.json_safety import make_json_safe
from app.services.semantic_schema import normalize_semantic_name


@dataclass(slots=True)
class DatasetResolution:
    dataset: ChatDataset | None
    candidates: list[ChatDataset]
    clarification: dict[str, Any] | None = None


def _is_role_allowed(dataset: ChatDataset, user: User | None) -> bool:
    if user is None:
        return True
    allowed_roles = set(dataset.allowed_roles or [])
    if not allowed_roles:
        return True
    return user.role.value in allowed_roles or user.role in {UserRole.manager, UserRole.admin} or user.role.value == "employee"


def _submission_access_allowed(submission: Submission, user: User | None) -> bool:
    if user is None:
        return True
    if user.role == UserRole.admin:
        return True
    if user.role == UserRole.employee:
        return submission.user_id == user.id
    if user.role == UserRole.manager:
        return bool(submission.user and submission.user.manager_id == user.id)
    return False


def _submission_dataset_key(submission: Submission) -> str:
    return f"job:{submission.id}"


def _submission_display_name(submission: Submission) -> str:
    instruction = str(submission.instruction or "").strip()
    if instruction:
        return instruction[:80] + ("..." if len(instruction) > 80 else "")
    if submission.file_name:
        return str(submission.file_name)
    return f"Submission {submission.id}"


def _submission_description(submission: Submission) -> str:
    parts = []
    if submission.file_name:
        parts.append(str(submission.file_name))
    if submission.output_format:
        parts.append(str(submission.output_format))
    if parts:
        return " - ".join(parts)
    return "Uploaded job dataset"


def _normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized = normalized.astype(object).where(pd.notnull(normalized), None)
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


async def _load_submission_frame(submission: Submission) -> pd.DataFrame:
    payload = submission.summary if isinstance(submission.summary, dict) else {}
    cleaned_data = payload.get("cleaned_data")
    if isinstance(cleaned_data, list) and cleaned_data and all(isinstance(item, dict) for item in cleaned_data):
        return _normalize_dataframe(pd.DataFrame(cleaned_data))

    structured_rows = getattr(submission, "structured_records", None)
    if isinstance(structured_rows, list) and structured_rows:
        rows = [row.payload for row in structured_rows if isinstance(getattr(row, "payload", None), dict)]
        if rows:
            return _normalize_dataframe(pd.DataFrame(rows))

    candidate_path = submission.output_path or submission.file_path
    if candidate_path:
        path = Path(candidate_path)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix in {".csv", ".tsv"}:
                frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
            elif suffix in {".xlsx", ".xls"}:
                frame = pd.read_excel(path)
            elif suffix == ".json":
                payload = pd.read_json(path)
                frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
            else:
                return pd.DataFrame()
            return _normalize_dataframe(frame)

    preview_rows = payload.get("preview_rows")
    if isinstance(preview_rows, list) and preview_rows and all(isinstance(item, dict) for item in preview_rows):
        return _normalize_dataframe(pd.DataFrame(preview_rows))

    return pd.DataFrame()


def _build_submission_columns(frame: pd.DataFrame, profile_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    profile_json = profile_json if isinstance(profile_json, dict) else {}
    profile_columns = profile_json.get("columns") if isinstance(profile_json.get("columns"), list) else []
    column_map = {
        str(column.get("name")): column
        for column in profile_columns
        if isinstance(column, dict) and column.get("name")
    }
    columns: list[dict[str, Any]] = []
    for column in frame.columns:
        profile = column_map.get(str(column), {})
        columns.append(
            {
                "physical_column": str(column),
                "semantic_name": str(profile.get("semantic_type_hint") or column).replace("_", " "),
                "description": str(profile.get("description") or ""),
                "data_type": str(profile.get("detected_type") or "string"),
                "sample_values": list(profile.get("sample_values") or []),
                "null_percentage": None,
                "distinct_count": profile.get("distinct_count"),
                "synonyms": list(profile.get("synonyms") or []),
                "sensitivity": str(profile.get("sensitivity") or "internal"),
            }
        )
    return columns


def _build_submission_sample_values(frame: pd.DataFrame) -> dict[str, list[str]]:
    sample_values: dict[str, list[str]] = {}
    for column in list(frame.columns)[:6]:
        series = frame[str(column)].dropna()
        sample_values[str(column)] = [str(value) for value in series.head(3).tolist()]
    return sample_values


async def ensure_submission_dataset(db: AsyncSession, submission: Submission) -> ChatDataset | None:
    dataset_key = _submission_dataset_key(submission)
    existing = (
        await db.execute(
            select(ChatDataset)
            .options(selectinload(ChatDataset.rows))
            .where(ChatDataset.dataset_key == dataset_key)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    frame = await _load_submission_frame(submission)
    if frame.empty:
        return None

    profile_payload: dict[str, Any] = {}
    if submission.data_profiles:
        latest_profile = max(
            (record for record in submission.data_profiles if isinstance(getattr(record, "profile_json", None), dict)),
            key=lambda record: str(getattr(record, "created_at", datetime.min.replace(tzinfo=UTC))),
            default=None,
        )
        if latest_profile is not None and isinstance(latest_profile.profile_json, dict):
            profile_payload = make_json_safe(latest_profile.profile_json)
    if not profile_payload and submission.file_path:
        built = build_data_profile_from_file(Path(str(submission.file_path)), max_preview_rows=max(25, min(len(frame), 50)))
        if built is not None:
            profile_payload = built[0]

    dataset = ChatDataset(
        dataset_key=dataset_key,
        source_submission_id=submission.id,
        display_name=_submission_display_name(submission),
        description=_submission_description(submission),
        domain="uploads",
        physical_source=str(submission.output_path or submission.file_path or "submission_rows"),
        storage_location=str(submission.output_path or submission.file_path or "submission_rows"),
        schema_version="1.0",
        columns_json=_build_submission_columns(frame, profile_payload),
        aliases=[str(submission.sub_id)] if submission.sub_id is not None else [],
        sample_values_json=_build_submission_sample_values(frame),
        owner=submission.user.full_name if submission.user else "",
        sensitivity="internal",
        allowed_roles=["employee", "manager", "admin"],
        row_level_policy_json={},
        column_level_policy_json={},
        chat_enabled=True,
        last_updated_at=submission.completed_at or submission.uploaded_at or datetime.now(UTC),
    )
    db.add(dataset)
    await db.flush()

    for index, (_, row) in enumerate(frame.head(5000).iterrows()):
        payload = make_json_safe({str(column): row[column] for column in frame.columns})
        db.add(ChatDatasetRow(dataset_id=dataset.id, row_index=index, payload=payload))

    await db.commit()
    await db.refresh(dataset)
    return dataset


async def ensure_submission_datasets(db: AsyncSession) -> None:
    submissions = (
        await db.execute(
            select(Submission)
            .options(
                selectinload(Submission.user),
                selectinload(Submission.data_profiles),
                selectinload(Submission.structured_records),
            )
            .order_by(Submission.uploaded_at.desc())
        )
    ).scalars().all()
    for submission in submissions:
        await ensure_submission_dataset(db, submission)


async def _accessible_submissions(db: AsyncSession, user: User | None) -> list[Submission]:
    stmt = (
        select(Submission)
        .options(
            selectinload(Submission.user),
            selectinload(Submission.data_profiles),
            selectinload(Submission.structured_records),
        )
        .order_by(Submission.uploaded_at.desc())
    )
    submissions = (await db.execute(stmt)).scalars().all()
    return [submission for submission in submissions if _submission_access_allowed(submission, user)]


async def list_chat_datasets(db: AsyncSession, user: User | None = None) -> list[ChatDataset]:
    await ensure_submission_datasets(db)
    stmt = (
        select(ChatDataset)
        .where(ChatDataset.chat_enabled.is_(True), ChatDataset.source_submission_id.is_not(None))
        .order_by(ChatDataset.last_updated_at.desc(), ChatDataset.display_name.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [dataset for dataset in rows if _is_role_allowed(dataset, user)]


def dataset_catalog_payload(dataset: ChatDataset) -> dict[str, Any]:
    return {
        "dataset_id": str(dataset.id),
        "dataset_key": dataset.dataset_key,
        "source_submission_id": str(dataset.source_submission_id) if getattr(dataset, "source_submission_id", None) else None,
        "display_name": dataset.display_name,
        "description": dataset.description,
        "domain": dataset.domain,
        "physical_source": dataset.physical_source,
        "storage_location": dataset.storage_location,
        "schema_version": dataset.schema_version,
        "aliases": list(dataset.aliases or []),
        "columns": list(dataset.columns_json or []),
        "sample_values": dataset.sample_values_json or {},
        "owner": dataset.owner,
        "sensitivity": dataset.sensitivity,
        "allowed_roles": list(dataset.allowed_roles or []),
        "chat_enabled": bool(dataset.chat_enabled),
        "last_updated_at": dataset.last_updated_at.isoformat() if dataset.last_updated_at else None,
    }


def dataset_to_frontend_payload(dataset: ChatDataset) -> dict[str, Any]:
    payload = dataset_catalog_payload(dataset)
    payload["columns"] = [
        {
            "physical_column": column.get("physical_column"),
            "semantic_name": column.get("semantic_name"),
            "description": column.get("description"),
            "data_type": column.get("data_type"),
            "sample_values": column.get("sample_values", []),
            "null_percentage": column.get("null_percentage"),
            "distinct_count": column.get("distinct_count"),
            "synonyms": column.get("synonyms", []),
            "sensitivity": column.get("sensitivity"),
        }
        for column in payload.get("columns", [])
        if isinstance(column, dict)
    ]
    return payload


async def load_dataset_rows(db: AsyncSession, dataset_id: UUID) -> list[dict[str, Any]]:
    stmt = (
        select(ChatDatasetRow)
        .where(ChatDatasetRow.dataset_id == dataset_id)
        .order_by(ChatDatasetRow.row_index.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [row.payload for row in rows if isinstance(row.payload, dict)]


async def load_dataset_frame(db: AsyncSession, dataset_id: UUID) -> pd.DataFrame:
    rows = await load_dataset_rows(db, dataset_id)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


async def resolve_dataset(
    db: AsyncSession,
    *,
    user: User | None,
    raw_reference: str,
) -> DatasetResolution:
    datasets = await list_chat_datasets(db, user)
    normalized = normalize_semantic_name(raw_reference)
    exact_matches: list[ChatDataset] = []
    alias_matches: list[ChatDataset] = []
    token_matches: list[ChatDataset] = []

    for dataset in datasets:
        terms = {
            normalize_semantic_name(dataset.dataset_key),
            normalize_semantic_name(dataset.display_name),
            *(normalize_semantic_name(alias) for alias in dataset.aliases or []),
        }
        if normalized in terms:
            exact_matches.append(dataset)
            continue
        if any(term and term in normalized for term in terms):
            alias_matches.append(dataset)
            continue
        if any(token and token in normalized for token in _dataset_tokens(dataset)):
            token_matches.append(dataset)

    if len(exact_matches) == 1:
        return DatasetResolution(dataset=exact_matches[0], candidates=exact_matches)
    if len(exact_matches) > 1:
        return DatasetResolution(
            dataset=None,
            candidates=exact_matches,
            clarification=_dataset_clarification(exact_matches),
        )
    candidates = alias_matches or token_matches
    if len(candidates) == 1:
        return DatasetResolution(dataset=candidates[0], candidates=candidates)
    if len(candidates) > 1:
        return DatasetResolution(dataset=None, candidates=candidates, clarification=_dataset_clarification(candidates))
    return DatasetResolution(dataset=None, candidates=[])


def _dataset_tokens(dataset: ChatDataset) -> set[str]:
    tokens = set(normalize_semantic_name(dataset.dataset_key).split("_"))
    tokens.update(normalize_semantic_name(dataset.display_name).split("_"))
    for alias in dataset.aliases or []:
        tokens.update(normalize_semantic_name(alias).split("_"))
    return {token for token in tokens if token}


def _dataset_clarification(candidates: list[ChatDataset]) -> dict[str, Any]:
    return {
        "question": "Which dataset should I use?",
        "options": [
            {
                "id": str(dataset.id),
                "label": dataset.display_name,
                "description": dataset.description,
            }
            for dataset in candidates
        ],
        "allow_free_text": False,
        "reason_code": "DATASET_AMBIGUOUS",
    }


def build_dataset_index(datasets: list[ChatDataset]) -> dict[str, ChatDataset]:
    index: dict[str, ChatDataset] = {}
    for dataset in datasets:
        index[str(dataset.id)] = dataset
        index[dataset.dataset_key] = dataset
        for alias in dataset.aliases or []:
            index[alias] = dataset
    return index
