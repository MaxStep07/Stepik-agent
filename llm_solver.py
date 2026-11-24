import re
import json
import time
from typing import Protocol, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from rich.console import Console

from config import Settings
from step_handlers import PreparedTask

T = TypeVar("T", bound=BaseModel)
console = Console()

SYSTEM_PROMPT = """Ты — эксперт по программированию, анализу данных (Python, Pandas, NumPy, SQL) и компьютерным наукам, решающий практические и теоретические задания на платформе Stepik.

Правила:
1. Внимательно опирайся на теорию текущего урока и точные формулировки условия.
2. Для задач по программированию (code):
   - Строго соблюдай имена переменных, функций, формат вывода (print vs return), типы данных и сортировку, указанные в условии или шаблоне.
   - Учитывай особенности Pandas (например, сброс индекса, обработку NaN, типы колонок, копирование срезов).
   - Не добавляй отладочные print(), если они не требуются задачей.
3. Если предыдущая попытка была отклонена грейдером Stepik, проанализируй текст ошибки/подсказки (`hint`), найди корневую причину и предложи исправленное решение, а не повторяй старое.
4. Отвечай строго в заданной JSON-схеме."""


class FatalLLMError(Exception):
    """Критическая ошибка LLM API (недоступны модели, неверный ключ, исчерпаны квоты)."""


class BaseSolver(Protocol):
    def start_conversation(
        self, task: PreparedTask, lesson_context: str
    ) -> "StepConversation": ...

    def generate_structured(
        self, messages: list[dict[str, str]], schema: type[T]
    ) -> tuple[T, str]: ...


class StepConversation:
    """Хранит диалог по одной задаче (универсально для любого провайдера)."""

    def __init__(
        self,
        solver: BaseSolver,
        task: PreparedTask,
        lesson_context: str,
    ) -> None:
        self.solver = solver
        self.task = task
        initial_text = (
            f"## Контекст урока\n{lesson_context}\n\n"
            f"---\n\n"
            f"## Текущее задание\n{task.prompt}"
        )
        self.messages: list[dict[str, str]] = [
            {"role": "user", "content": initial_text}
        ]

    def ask(self) -> BaseModel:
        parsed, raw_json = self.solver.generate_structured(
            messages=self.messages,
            schema=self.task.response_schema,
        )
        self.messages.append({"role": "assistant", "content": raw_json})
        return parsed

    def add_feedback(self, status: str, hint: str) -> None:
        feedback = (
            f"Грейдер Stepik отклонил предыдущий ответ.\n"
            f"Статус: {status}\n"
            f"Сообщение грейдера (hint):\n{hint or '(без подробностей)'}\n\n"
            f"Проанализируй ошибку и выдай исправленное решение в той же JSON-схеме."
        )
        self.messages.append({"role": "user", "content": feedback})


def _extract_json(text: str) -> str:
    """Убирает блок рассуждений <think> и извлекает JSON-объект."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


class GeminiSolver:
    def __init__(self, settings: Settings) -> None:
        self.models = list(
            dict.fromkeys([settings.gemini_model, *settings.gemini_fallback_models])
        )
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )

    def start_conversation(
        self, task: PreparedTask, lesson_context: str
    ) -> StepConversation:
        return StepConversation(self, task, lesson_context)

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        retries_per_model: int = 2,
    ) -> tuple[T, str]:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )

        last_error: Exception | None = None
        for model_name in self.models:
            for attempt in range(1, retries_per_model + 1):
                t0 = time.perf_counter()
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    elapsed = time.perf_counter() - t0
                    console.print(f"      [dim]{model_name} ответил за {elapsed:.1f}с[/dim]")

                    raw_text = response.text or ""
                    if response.parsed is not None and isinstance(response.parsed, schema):
                        return response.parsed, raw_text
                    parsed = schema.model_validate_json(_extract_json(raw_text))
                    return parsed, raw_text

                except genai_errors.APIError as exc:
                    last_error = exc
                    code = getattr(exc, "code", None)
                    if code == 404:
                        console.print(f"      [yellow]{model_name}: 404, пробую следующую...[/yellow]")
                        break
                    if code in (429, 500, 503, 504):
                        console.print(
                            f"      [yellow]{model_name}: HTTP {code} "
                            f"(попытка {attempt}/{retries_per_model})[/yellow]"
                        )
                        if attempt < retries_per_model:
                            time.sleep(4.0)
                            continue
                        break
                    raise FatalLLMError(str(exc)) from exc
                except Exception as exc:
                    last_error = exc
                    if attempt < retries_per_model:
                        time.sleep(3.0)
                        continue
                    break

        raise FatalLLMError(f"Все модели Gemini недоступны: {last_error}")


class OpenAICompatibleSolver:
    """Работает с Groq, OpenRouter, DeepSeek, Cerebras, Ollama и любым OpenAI-совместимым API."""
    
    def __init__(self, settings: Settings) -> None:
        self.models = list(
            dict.fromkeys([settings.openai_model, *settings.openai_fallback_models])
        )
        self.client = httpx.Client(
            base_url=settings.openai_base_url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://stepik.org",
                "X-Title": "Stepik Agent",
            },
            timeout=60.0,
        )

    def start_conversation(
        self, task: PreparedTask, lesson_context: str
    ) -> StepConversation:
        return StepConversation(self, task, lesson_context)

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        retries_per_model: int = 2,
    ) -> tuple[T, str]:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Верни ТОЛЬКО валидный JSON-объект без пояснений вокруг, строго по схеме:\n"
            f"{schema_json}"
        )
        full_messages = [{"role": "system", "content": system_content}, *messages]

        last_error: Exception | None = None
        for model_name in self.models:
            use_json_mode = True
            for attempt in range(1, retries_per_model + 1):
                t0 = time.perf_counter()
                payload: dict = {
                    "model": model_name,
                    "messages": full_messages,
                    "temperature": 0.2,
                }
                if use_json_mode:
                    payload["response_format"] = {"type": "json_object"}

                try:
                    resp = self.client.post("/chat/completions", json=payload)
                except httpx.RequestError as exc:
                    last_error = exc
                    console.print(f"      [yellow]{model_name}: сетевая ошибка ({exc})[/yellow]")
                    if attempt < retries_per_model:
                        time.sleep(3.0)
                        continue
                    break

                # Некоторые модели не поддерживают response_format: json_object
                if resp.status_code == 400 and use_json_mode and any(
                    k in resp.text.lower() for k in ("response_format", "json_validate_failed", "json")
                ):
                    use_json_mode = False
                    continue

                if resp.status_code in (401, 403):
                    raise FatalLLMError(
                        f"Ошибка авторизации LLM API HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                if resp.status_code == 404:
                    last_error = RuntimeError(f"404: {resp.text[:200]}")
                    console.print(f"      [yellow]{model_name}: 404 (модель не найдена)[/yellow]")
                    break

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    console.print(
                        f"      [yellow]{model_name}: HTTP {resp.status_code} "
                        f"(попытка {attempt}/{retries_per_model})[/yellow]"
                    )
                    if attempt < retries_per_model:
                        retry_after = resp.headers.get("Retry-After")
                        wait = min(float(retry_after), 15.0) if retry_after and retry_after.replace(".", "", 1).isdigit() else 4.0
                        time.sleep(wait)
                        continue
                    break

                if resp.status_code >= 400:
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    console.print(f"      [yellow]{model_name}: HTTP {resp.status_code}[/yellow]")
                    break

                elapsed = time.perf_counter() - t0
                console.print(f"      [dim]{model_name} ответил за {elapsed:.1f}с[/dim]")

                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"] or ""
                try:
                    cleaned = _extract_json(raw_text)
                    parsed = schema.model_validate_json(cleaned)
                    return parsed, cleaned
                except Exception as exc:
                    last_error = exc
                    console.print(
                        f"      [yellow]{model_name}: невалидный JSON ({exc})[/yellow]"
                    )
                    if attempt < retries_per_model:
                        continue
                    break

        raise FatalLLMError(f"Все OpenAI-совместимые модели недоступны: {last_error}")


def create_solver(settings: Settings) -> BaseSolver:
    if settings.llm_provider == "openai":
        return OpenAICompatibleSolver(settings)
    return GeminiSolver(settings)