import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_markdown(html: str | None) -> str:
    """Очищает HTML со Stepik и переводит в компактный Markdown."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()

    text = md(
        str(soup),
        heading_style="ATX",
        code_language="",
        strip=["img"],
    )
    # Убираем лишние пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LessonContext:
    """Хранит теоретический контекст в рамках одного урока с ограничением по длине."""

    def __init__(self, lesson_title: str, max_chars: int = 12_000) -> None:
        self.lesson_title = lesson_title
        self.max_chars = max_chars
        self._chunks: list[str] = []

    def add_theory(self, step_position: int, html: str | None) -> None:
        cleaned = html_to_markdown(html)
        if cleaned:
            self._chunks.append(f"### Теория (шаг {step_position})\n{cleaned}")

    def render(self) -> str:
        if not self._chunks:
            return f"Урок: {self.lesson_title}\n(Теоретических шагов перед задачей нет.)"

        combined = "\n\n".join(self._chunks)
        if len(combined) > self.max_chars:
            # Оставляем самую свежую теорию ближе к текущему шагу
            combined = "...[начало теории сокращено]...\n\n" + combined[-self.max_chars :]

        return f"Урок: {self.lesson_title}\n\n{combined}"