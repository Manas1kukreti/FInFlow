from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    app_name: str = "LedgerFlow Analytics API"
    database_url: str = "postgresql+asyncpg://ledgerflow:ledgerflow@localhost:5433/ledgerflow"
    redis_url: str = "redis://localhost:6379/0"
    agent_base_url: str = "http://localhost:8001"
    backend_callback_url: str = "http://localhost:8000/api/agent/callback"
    agent_callback_secret: str = "change-agent-callback-secret"
    agent_registry_secret: str = "change-agent-registry-secret"
    enable_needs_review_jobs: bool = True
    agent_service_secret: str = "change-agent-service-secret"
    agent_dispatch_queue: str = "finflow:agent:dispatch"
    agent_dispatch_retry_queue: str = "finflow:agent:dispatch:retry"
    agent_dead_letter_queue: str = "finflow:agent:dispatch:dead"
    dispatch_max_attempts: int = 3
    dispatch_retry_base_delay_seconds: int = 15
    dispatch_retry_max_delay_seconds: int = 300
    upload_dir: str = "./storage/uploads"
    output_dir: str = "./storage/outputs"
    max_preview_rows: int = 50
    max_upload_size_mb: int = 10
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    trusted_proxy_ips: str = ""
    frontend_base_url: str = "http://localhost:5173"
    default_admin_email: str = "admin@example.com"
    default_admin_password: str = "ChangeMeAdmin123"
    default_admin_name: str = "LedgerFlow Admin"
    seed_default_users: bool = False
    agent_email: str | None = None
    agent_password: str | None = None
    agent_name: str = "LedgerFlow Posting Agent"
    emails_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str = "no-reply@ledgerflow.local"
    smtp_tls: bool = True
    display_timezone: str = "Asia/Kolkata"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "Settings":
        self.database_url = normalize_database_url(self.database_url)
        if self.environment.lower() in {"prod", "production"}:
            validate_production_settings(self)
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def trusted_proxy_ip_list(self) -> list[str]:
        return [ip.strip() for ip in self.trusted_proxy_ips.split(",") if ip.strip()]

    @property
    def display_tzinfo(self):
        try:
            return ZoneInfo(self.display_timezone)
        except ZoneInfoNotFoundError:
            return ZoneInfo("UTC")


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


def validate_production_settings(settings: Settings) -> None:
    unsafe_jwt_secrets = {"change-me-in-production", "change-this-before-production"}
    if settings.jwt_secret_key in unsafe_jwt_secrets or len(settings.jwt_secret_key) < 32:
        raise ValueError("JWT_SECRET_KEY must be a unique production secret of at least 32 characters")
    if settings.agent_callback_secret == "change-agent-callback-secret":
        raise ValueError("AGENT_CALLBACK_SECRET must be set to the shared backend/agent callback secret")
    if settings.default_admin_password == "ChangeMeAdmin123" or len(settings.default_admin_password) < 12:
        raise ValueError("DEFAULT_ADMIN_PASSWORD must be changed before production deploy")
    if bool(settings.agent_email) != bool(settings.agent_password):
        raise ValueError("AGENT_EMAIL and AGENT_PASSWORD must be set together")
    if settings.agent_password and len(settings.agent_password) < 12:
        raise ValueError("AGENT_PASSWORD must be at least 12 characters in production")
    if "localhost" in settings.database_url or "127.0.0.1" in settings.database_url:
        raise ValueError("DATABASE_URL must point to the production database")
    if "localhost" in settings.agent_base_url or "127.0.0.1" in settings.agent_base_url:
        raise ValueError("AGENT_BASE_URL must point to the production agent service")
    if "*" in settings.cors_origin_list:
        raise ValueError("CORS_ORIGINS must list exact frontend origins in production")
    if any("localhost" in origin or "127.0.0.1" in origin for origin in settings.cors_origin_list):
        raise ValueError("CORS_ORIGINS must not include localhost in production")


@lru_cache
def get_settings() -> Settings:
    return Settings()
