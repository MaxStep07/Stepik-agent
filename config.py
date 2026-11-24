import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Отсутствует обязательная переменная окружения: {name}")
    return value


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "gemini" или "openai"
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_models: tuple[str, ...]
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_fallback_models: tuple[str, ...]
    stepik_session_id: str
    stepik_csrf_token: str
    course_id: int
    max_attempts_per_step: int
    min_delay_sec: float
    max_delay_sec: float
    submission_poll_interval: float = 2.0
    submission_timeout_sec: float = 90.0

    @property
    def active_model(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.gemini_model


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in ("gemini", "openai"):
        raise ValueError(f"Неизвестный LLM_PROVIDER: {provider} (используй 'gemini' или 'openai')")

    gemini_key = _require_env("GEMINI_API_KEY") if provider == "gemini" else os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = _require_env("OPENAI_API_KEY") if provider == "openai" else os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = (
        _require_env("OPENAI_BASE_URL").rstrip("/")
        if provider == "openai"
        else os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    )
    openai_model = (
        _require_env("OPENAI_MODEL")
        if provider == "openai"
        else os.getenv("OPENAI_MODEL", "").strip()
    )

    return Settings(
        llm_provider=provider,
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        gemini_fallback_models=_parse_csv(os.getenv("GEMINI_FALLBACK_MODELS", "")),
        openai_api_key=openai_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_fallback_models=_parse_csv(os.getenv("OPENAI_FALLBACK_MODELS", "")),
        stepik_session_id=_require_env("STEPIK_SESSION_ID"),
        stepik_csrf_token=_require_env("STEPIK_CSRF_TOKEN"),
        course_id=int(_require_env("STEPIK_COURSE_ID")),
        max_attempts_per_step=int(os.getenv("MAX_ATTEMPTS_PER_STEP", "4")),
        min_delay_sec=float(os.getenv("MIN_DELAY_SEC", "1.5")),
        max_delay_sec=float(os.getenv("MAX_DELAY_SEC", "3.5")),
    )