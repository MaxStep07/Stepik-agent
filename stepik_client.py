import random
import time
from typing import Any

import httpx

from config import Settings


class StepikAPIError(Exception):
    """Ошибка взаимодействия со Stepik API."""


class StepikClient:
    BASE_URL = "https://stepik.org/api"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(
            base_url=self.BASE_URL,
            cookies={
                "sessionid": settings.stepik_session_id,
                "csrftoken": settings.stepik_csrf_token,
            },
            headers={
                "X-CSRFToken": settings.stepik_csrf_token,
                "Referer": "https://stepik.org/",
                "Origin": "https://stepik.org",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "StepikClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def polite_delay(self) -> None:
        delay = random.uniform(
            self.settings.min_delay_sec,
            self.settings.max_delay_sec,
        )
        time.sleep(delay)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: list[tuple[str, Any]] | dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> dict[str, Any]:
        backoff = 2.0
        for attempt in range(1, retries + 1):
            try:
                response = self.client.request(
                    method, endpoint, params=params, json=json
                )
            except httpx.RequestError as exc:
                if attempt == retries:
                    raise StepikAPIError(f"Сетевая ошибка ({endpoint}): {exc}") from exc
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == retries:
                    raise StepikAPIError(
                        f"HTTP {response.status_code} на {endpoint}: {response.text[:300]}"
                    )
                retry_after = response.headers.get("Retry-After")
                sleep_time = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                time.sleep(sleep_time)
                backoff *= 2
                continue

            if response.status_code in (401, 403):
                raise StepikAPIError(
                    f"Ошибка авторизации HTTP {response.status_code} ({endpoint}). "
                    f"Проверь sessionid и csrftoken в .env. Детали: {response.text[:300]}"
                )

            if response.status_code >= 400:
                raise StepikAPIError(
                    f"Ошибка HTTP {response.status_code} ({endpoint}): {response.text[:400]}"
                )

            return response.json() if response.content else {}

        raise StepikAPIError(f"Превышено число попыток запроса к {endpoint}")

    # --- Авторизация и загрузка структуры курса ---

    def get_current_user(self) -> dict[str, Any]:
        data = self._request("GET", "/stepics/1")
        users = data.get("users", [])
        if not users or users[0].get("is_guest", True):
            raise StepikAPIError(
                "Сессия не авторизована (определен как гость). Обнови sessionid в .env."
            )
        return users[0]

    def get_objects(
        self,
        resource: str,
        ids: list[int | str],
        chunk_size: int = 50,
    ) -> list[dict[str, Any]]:
        """Загружает объекты пачками и возвращает их в точном порядке переданных ids."""
        if not ids:
            return []

        unique_ids = list(dict.fromkeys(ids))
        by_id: dict[str, dict[str, Any]] = {}

        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i : i + chunk_size]
            params = [("ids[]", item_id) for item_id in chunk]
            data = self._request("GET", f"/{resource}", params=params)
            for obj in data.get(resource, []):
                by_id[str(obj["id"])] = obj

        return [by_id[str(item_id)] for item_id in ids if str(item_id) in by_id]

    def get_course(self, course_id: int) -> dict[str, Any]:
        courses = self.get_objects("courses", [course_id])
        if not courses:
            raise StepikAPIError(f"Курс {course_id} не найден или нет доступа.")
        return courses[0]

    def get_course_tree(self, course_id: int) -> dict[str, Any]:
        """
        Возвращает полную структуру курса:
        sections -> units -> lessons -> steps (с флагом is_passed и assignment_id).
        """
        course = self.get_course(course_id)
        sections = self.get_objects("sections", course.get("sections", []))

        all_unit_ids = [uid for s in sections for uid in s.get("units", [])]
        units = self.get_objects("units", all_unit_ids)
        units_by_id = {u["id"]: u for u in units}

        all_lesson_ids = [u["lesson"] for u in units if u.get("lesson")]
        lessons = self.get_objects("lessons", all_lesson_ids)
        lessons_by_id = {l["id"]: l for l in lessons}

        all_assignment_ids = [aid for u in units for aid in u.get("assignments", [])]
        assignments = self.get_objects("assignments", all_assignment_ids)
        # Маппинг (unit_id, step_id) -> assignment_id
        assignment_map: dict[tuple[int, int], int] = {
            (a["unit"], a["step"]): a["id"] for a in assignments
        }

        all_step_ids = [sid for l in lessons for sid in l.get("steps", [])]
        steps = self.get_objects("steps", all_step_ids)
        steps_by_id = {st["id"]: st for st in steps}

        progress_ids = [st["progress"] for st in steps if st.get("progress")]
        progresses = self.get_objects("progresses", progress_ids)
        progress_by_id = {p["id"]: p for p in progresses}

        tree_sections = []
        for section in sections:
            section_lessons = []
            for unit_id in section.get("units", []):
                unit = units_by_id.get(unit_id)
                if not unit:
                    continue
                lesson = lessons_by_id.get(unit["lesson"])
                if not lesson:
                    continue

                lesson_steps = []
                for step_id in lesson.get("steps", []):
                    step = steps_by_id.get(step_id)
                    if not step:
                        continue
                    progress = progress_by_id.get(step.get("progress", ""), {})
                    lesson_steps.append(
                        {
                            "id": step["id"],
                            "position": step["position"],
                            "lesson_id": lesson["id"],
                            "unit_id": unit["id"],
                            "assignment_id": assignment_map.get((unit["id"], step["id"])),
                            "block": step.get("block", {}),
                            "is_passed": bool(progress.get("is_passed", False)),
                            "url": (
                                f"https://stepik.org/lesson/{lesson['id']}"
                                f"/step/{step['position']}?unit={unit['id']}"
                            ),
                        }
                    )

                section_lessons.append(
                    {
                        "unit_id": unit["id"],
                        "lesson_id": lesson["id"],
                        "title": lesson.get("title", f"Lesson {lesson['id']}"),
                        "steps": lesson_steps,
                    }
                )

            tree_sections.append(
                {
                    "id": section["id"],
                    "position": section.get("position", 0),
                    "title": section.get("title", f"Section {section['id']}"),
                    "lessons": section_lessons,
                }
            )

        return {
            "id": course["id"],
            "title": course.get("title", f"Course {course_id}"),
            "sections": tree_sections,
        }

    # --- Действия с шагами: просмотры, попытки, отправка решений ---

    def mark_step_viewed(self, step_id: int, assignment_id: int | None) -> None:
        payload = {"view": {"step": step_id, "assignment": assignment_id}}
        self._request("POST", "/views", json=payload)

    def create_attempt(self, step_id: int) -> dict[str, Any]:
        """Создает новую попытку (или получает активную) и ждет генерации датасета."""
        data = self._request("POST", "/attempts", json={"attempt": {"step": step_id}})
        attempt = data["attempts"][0]
        attempt_id = attempt["id"]

        deadline = time.time() + 30.0
        while attempt.get("status") != "active" and time.time() < deadline:
            time.sleep(1.0)
            poll = self._request("GET", f"/attempts/{attempt_id}")
            attempt = poll["attempts"][0]

        if attempt.get("status") != "active":
            raise StepikAPIError(
                f"Попытка {attempt_id} для шага {step_id} не перешла в статус active: {attempt}"
            )
        return attempt

    def submit_and_wait(
        self,
        attempt_id: int,
        reply: dict[str, Any],
    ) -> dict[str, Any]:
        """Отправляет ответ и опрашивает грейдер до финального статуса."""
        payload = {
            "submission": {
                "attempt": attempt_id,
                "reply": reply,
            }
        }
        data = self._request("POST", "/submissions", json=payload)
        submission = data["submissions"][0]
        submission_id = submission["id"]

        deadline = time.time() + self.settings.submission_timeout_sec
        while submission.get("status") == "evaluation" and time.time() < deadline:
            time.sleep(self.settings.submission_poll_interval)
            poll = self._request("GET", f"/submissions/{submission_id}")
            submission = poll["submissions"][0]

        return submission