from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from context_builder import html_to_markdown


# --- Pydantic-схемы для Structured Outputs Gemini ---


class CodeAnswer(BaseModel):
    reasoning: str = Field(description="Краткий план решения и проверка крайних случаев.")
    code: str = Field(
        description="Полный исходный код для отправки в редактор Stepik (без обертки ```)."
    )


class SqlAnswer(BaseModel):
    reasoning: str = Field(description="Краткое объяснение SQL-запроса.")
    solve_sql: str = Field(description="Чистый SQL-запрос без markdown-разметки.")


class ChoiceAnswer(BaseModel):
    reasoning: str = Field(description="Краткий разбор вариантов.")
    selected_indices: list[int] = Field(
        description="Список 0-based индексов правильных вариантов ответа."
    )


class StringAnswer(BaseModel):
    reasoning: str = Field(description="Краткое обоснование ответа.")
    text: str = Field(description="Точная строка ответа без лишних кавычек и пояснений.")


class NumberAnswer(BaseModel):
    reasoning: str = Field(description="Краткий расчет.")
    number: str = Field(
        description="Числовой ответ в виде строки (например '42' или '3.14')."
    )


class MathAnswer(BaseModel):
    reasoning: str = Field(description="Краткий вывод формулы.")
    formula: str = Field(description="Математическое выражение в формате Stepik.")


class OrderingAnswer(BaseModel):
    reasoning: str = Field(description="Краткое обоснование порядка.")
    ordering: list[int] = Field(
        description="Перестановка 0-based индексов от 0 до N-1."
    )


class FillBlanksAnswer(BaseModel):
    reasoning: str = Field(description="Краткое обоснование каждого пропуска.")
    blanks: list[str] = Field(
        description="Список строк-ответов строго по порядку пропусков."
    )


class TableRowSelection(BaseModel):
    row_index: int = Field(description="0-based индекс строки.")
    selected_column_indices: list[int] = Field(
        description="0-based индексы выбранных столбцов для этой строки."
    )


class TableAnswer(BaseModel):
    reasoning: str = Field(description="Краткое обоснование выбора ячеек.")
    rows: list[TableRowSelection] = Field(description="Выбор по каждой строке таблицы.")


# --- Подготовленное задание ---


@dataclass
class PreparedTask:
    step_type: str
    prompt: str
    response_schema: type[BaseModel]
    metadata: dict[str, Any]


SUPPORTED_SOLVABLE_TYPES = {
    "code",
    "sql",
    "choice",
    "string",
    "number",
    "math",
    "sorting",
    "matching",
    "fill-blanks",
    "table",
}

THEORY_TYPES = {"text", "video"}


def _pick_language(templates: dict[str, str], preferred: str = "python3") -> str:
    if not templates:
        return preferred
    if preferred in templates:
        return preferred
    for lang in templates:
        if lang.startswith("python"):
            return lang
    return next(iter(templates))


def prepare_task(step: dict[str, Any], attempt: dict[str, Any]) -> PreparedTask:
    """Формирует промпт, схему ответа и метаданные под конкретный тип шага."""
    block = step.get("block", {})
    step_type = block.get("name", "unknown")
    statement = html_to_markdown(block.get("text", ""))
    options = block.get("options") or {}
    dataset = attempt.get("dataset") or {}

    if step_type == "code":
        templates: dict[str, str] = options.get("code_templates") or {}
        language = _pick_language(templates)
        starter_code = templates.get(language, "")
        samples: list[list[str]] = options.get("samples") or []

        parts = [
            f"Тип задания: написание кода (язык: `{language}`).",
            f"## Условие\n{statement}",
        ]
        if starter_code.strip():
            parts.append(
                f"## Стартовый шаблон редактора\n"
                f"Сохрани сигнатуры функций, имена переменных и структуру шаблона, если они заданы.\n"
                f"```{language}\n{starter_code}\n```"
            )
        if samples:
            sample_lines = []
            for idx, sample in enumerate(samples, 1):
                inp = sample[0] if len(sample) > 0 else ""
                out = sample[1] if len(sample) > 1 else ""
                sample_lines.append(
                    f"Пример {idx}:\nВвод:\n```\n{inp}\n```\nВывод:\n```\n{out}\n```"
                )
            parts.append("## Примеры тестов\n" + "\n\n".join(sample_lines))

        parts.append(
            "Верни готовый рабочий код в поле `code`. Не добавляй markdown-ограждения ```."
        )
        return PreparedTask(
            step_type=step_type,
            prompt="\n\n".join(parts),
            response_schema=CodeAnswer,
            metadata={"language": language},
        )

    if step_type == "sql":
        prompt = (
            f"Тип задания: SQL-запрос.\n\n## Условие\n{statement}\n\n"
            f"Верни чистый SQL-запрос в поле `solve_sql`."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=SqlAnswer,
            metadata={},
        )

    if step_type == "choice":
        is_multiple = bool(dataset.get("is_multiple_choice", False))
        raw_options: list[str] = dataset.get("options") or []
        formatted = "\n".join(
            f"[{i}] {html_to_markdown(opt)}" for i, opt in enumerate(raw_options)
        )
        mode_text = (
            "Множественный выбор (выбери ВСЕ верные варианты)."
            if is_multiple
            else "Одиночный выбор (выбери РОВНО ОДИН верный вариант)."
        )
        prompt = (
            f"Тип задания: тест ({mode_text})\n\n"
            f"## Условие\n{statement}\n\n"
            f"## Варианты (индексация с 0)\n{formatted}"
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=ChoiceAnswer,
            metadata={
                "num_options": len(raw_options),
                "is_multiple": is_multiple,
            },
        )

    if step_type == "string":
        prompt = (
            f"Тип задания: краткий текстовый ответ.\n\n## Условие\n{statement}\n\n"
            f"Укажи точный ответ в поле `text`."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=StringAnswer,
            metadata={},
        )

    if step_type == "number":
        prompt = (
            f"Тип задания: числовой ответ.\n\n## Условие\n{statement}\n\n"
            f"Укажи только число в поле `number` (десятичный разделитель — точка)."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=NumberAnswer,
            metadata={},
        )

    if step_type == "math":
        prompt = (
            f"Тип задания: математическая формула.\n\n## Условие\n{statement}\n\n"
            f"Запиши формулу в поле `formula` в стандартной нотации (как в Python/SymPy)."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=MathAnswer,
            metadata={},
        )

    if step_type == "sorting":
        raw_options = dataset.get("options") or []
        formatted = "\n".join(
            f"[{i}] {html_to_markdown(opt)}" for i, opt in enumerate(raw_options)
        )
        prompt = (
            f"Тип задания: сортировка элементов в правильном порядке.\n\n"
            f"## Условие\n{statement}\n\n"
            f"## Элементы (индексация с 0)\n{formatted}\n\n"
            f"Верни в `ordering` перестановку индексов от 0 до {len(raw_options) - 1} "
            f"в нужном порядке."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=OrderingAnswer,
            metadata={"count": len(raw_options)},
        )

    if step_type == "matching":
        pairs: list[dict[str, str]] = dataset.get("pairs") or []
        left_col = "\n".join(
            f"[{i}] {html_to_markdown(p.get('first', ''))}" for i, p in enumerate(pairs)
        )
        right_col = "\n".join(
            f"[{i}] {html_to_markdown(p.get('second', ''))}" for i, p in enumerate(pairs)
        )
        prompt = (
            f"Тип задания: сопоставление пар.\n\n"
            f"## Условие\n{statement}\n\n"
            f"## Левый столбец (порядок фиксирован)\n{left_col}\n\n"
            f"## Правый столбец (нужно подобрать индекс для каждой строки слева)\n{right_col}\n\n"
            f"Верни `ordering` длины {len(pairs)}, где `ordering[i]` — индекс элемента "
            f"из правого столбца, соответствующий элементу `i` из левого."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=OrderingAnswer,
            metadata={"count": len(pairs)},
        )

    if step_type == "fill-blanks":
        components: list[dict[str, Any]] = dataset.get("components") or []
        rendered_parts = []
        blank_idx = 0
        blank_meta = []
        for comp in components:
            ctype = comp.get("type")
            if ctype == "text":
                rendered_parts.append(html_to_markdown(comp.get("text", "")))
            elif ctype == "input":
                rendered_parts.append(f" [ПРОПУСК #{blank_idx}: свободный ввод] ")
                blank_meta.append({"type": "input", "options": []})
                blank_idx += 1
            elif ctype == "select":
                opts = comp.get("options") or []
                rendered_parts.append(
                    f" [ПРОПУСК #{blank_idx}: выбери ровно один из {opts}] "
                )
                blank_meta.append({"type": "select", "options": opts})
                blank_idx += 1

        prompt = (
            f"Тип задания: заполнение пропусков.\n\n"
            f"## Условие\n{statement}\n\n"
            f"## Текст с пропусками\n{''.join(rendered_parts)}\n\n"
            f"Верни в `blanks` ровно {blank_idx} строк по порядку пропусков."
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=FillBlanksAnswer,
            metadata={"blanks": blank_meta},
        )

    if step_type == "table":
        rows: list[str] = dataset.get("rows") or []
        columns: list[str] = dataset.get("columns") or []
        is_checkbox = bool(dataset.get("is_checkbox", False))
        mode = "можно выбрать несколько столбцов в строке" if is_checkbox else "ровно один столбец в каждой строке"

        rows_txt = "\n".join(f"[{i}] {html_to_markdown(r)}" for i, r in enumerate(rows))
        cols_txt = "\n".join(f"[{i}] {html_to_markdown(c)}" for i, c in enumerate(columns))
        prompt = (
            f"Тип задания: таблица ({mode}).\n\n"
            f"## Условие\n{statement}\n\n"
            f"## Строки (индексация с 0)\n{rows_txt}\n\n"
            f"## Столбцы (индексация с 0)\n{cols_txt}"
        )
        return PreparedTask(
            step_type=step_type,
            prompt=prompt,
            response_schema=TableAnswer,
            metadata={"rows": rows, "columns": columns, "is_checkbox": is_checkbox},
        )

    raise ValueError(f"Неподдерживаемый тип шага: {step_type}")


def build_reply(task: PreparedTask, answer: BaseModel) -> dict[str, Any]:
    """Преобразует валидированный ответ LLM в JSON-формат Stepik API."""
    t = task.step_type
    meta = task.metadata

    if t == "code":
        assert isinstance(answer, CodeAnswer)
        code = answer.code.strip()
        # На случай если модель всё-таки обернула код в ```python ... ```
        if code.startswith("```"):
            lines = code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return {"language": meta["language"], "code": code}

    if t == "sql":
        assert isinstance(answer, SqlAnswer)
        return {"solve_sql": answer.solve_sql.strip()}

    if t == "choice":
        assert isinstance(answer, ChoiceAnswer)
        n = meta["num_options"]
        valid_indices = [i for i in answer.selected_indices if 0 <= i < n]
        if not meta["is_multiple"] and len(valid_indices) > 1:
            valid_indices = [valid_indices[0]]
        choices = [i in valid_indices for i in range(n)]
        return {"choices": choices}

    if t == "string":
        assert isinstance(answer, StringAnswer)
        return {"text": answer.text.strip()}

    if t == "number":
        assert isinstance(answer, NumberAnswer)
        return {"number": answer.number.strip().replace(",", ".")}

    if t == "math":
        assert isinstance(answer, MathAnswer)
        return {"formula": answer.formula.strip()}

    if t in ("sorting", "matching"):
        assert isinstance(answer, OrderingAnswer)
        count = meta["count"]
        ordering = answer.ordering
        if sorted(ordering) != list(range(count)):
            raise ValueError(
                f"Некорректная перестановка {ordering} для {count} элементов."
            )
        return {"ordering": ordering}

    if t == "fill-blanks":
        assert isinstance(answer, FillBlanksAnswer)
        expected = len(meta["blanks"])
        if len(answer.blanks) != expected:
            raise ValueError(
                f"Ожидалось {expected} пропусков, получено {len(answer.blanks)}."
            )
        return {"blanks": [b.strip() for b in answer.blanks]}

    if t == "table":
        assert isinstance(answer, TableAnswer)
        rows: list[str] = meta["rows"]
        columns: list[str] = meta["columns"]
        selected_map = {
            r.row_index: set(r.selected_column_indices) for r in answer.rows
        }
        choices_payload = []
        for r_idx, row_name in enumerate(rows):
            sel_cols = selected_map.get(r_idx, set())
            choices_payload.append(
                {
                    "name_row": row_name,
                    "columns": [
                        {"name": col_name, "answer": c_idx in sel_cols}
                        for c_idx, col_name in enumerate(columns)
                    ],
                }
            )
        return {"choices": choices_payload}

    raise ValueError(f"Невозможно собрать reply для типа {t}")