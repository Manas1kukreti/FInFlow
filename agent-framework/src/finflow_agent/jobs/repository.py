from abc import ABC, abstractmethod
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseJobRepository(ABC):
    """
    Abstract interface for JobRepository implementations.
    """
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def create_or_update_queued(self, job_id: str, submission_id: str, payload: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def mark_planning(self, job_id: str) -> None:
        pass

    @abstractmethod
    async def mark_running(self, job_id: str) -> None:
        pass

    @abstractmethod
    async def mark_succeeded(self, job_id: str, result: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def mark_failed(self, job_id: str, error_msg: str) -> None:
        pass

    @abstractmethod
    async def mark_quarantined(self, job_id: str, reason: str) -> None:
        pass

    @abstractmethod
    async def mark_callback_failed(self, job_id: str, error_msg: Optional[str] = None) -> None:
        pass

class JobRepository(BaseJobRepository):
    """
    This is a local-development repository only. It is not safe for concurrent production workers. 
    Replace with PostgreSQL-backed implementation before production.

    TODO: Implement PostgreSQL-backed JobRepository using SQLAlchemy/asyncpg to support concurrent 
    production worker operations.
    """
    def __init__(self, db_path: str = "jobs_store.json"):
        # Put in same directory as process or a temp directory
        self.db_path = os.path.abspath(db_path)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def _read_db(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.db_path):
                return {}
            with open(self.db_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading jobs DB: {e}")
            return {}

    def _write_db(self, data: Dict[str, Any]):
        try:
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing jobs DB: {e}")

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        db = self._read_db()
        return db.get(job_id)

    async def create_or_update_queued(self, job_id: str, submission_id: str, payload: Dict[str, Any]) -> None:
        db = self._read_db()
        db[job_id] = {
            "job_id": job_id,
            "submission_id": submission_id,
            "status": "QUEUED",
            "payload": payload,
            "result": None,
            "error": None
        }
        self._write_db(db)

    async def mark_planning(self, job_id: str) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "PLANNING"
            self._write_db(db)

    async def mark_running(self, job_id: str) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "RUNNING"
            self._write_db(db)

    async def mark_succeeded(self, job_id: str, result: Dict[str, Any]) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "SUCCEEDED"
            db[job_id]["result"] = result
            self._write_db(db)

    async def mark_failed(self, job_id: str, error_msg: str) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "FAILED"
            db[job_id]["error"] = error_msg
            self._write_db(db)

    async def mark_quarantined(self, job_id: str, reason: str) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "QUARANTINED"
            db[job_id]["error"] = reason
            self._write_db(db)

    async def mark_callback_failed(self, job_id: str, error_msg: Optional[str] = None) -> None:
        db = self._read_db()
        if job_id in db:
            db[job_id]["status"] = "CALLBACK_FAILED"
            if error_msg:
                db[job_id]["error"] = error_msg
            self._write_db(db)
