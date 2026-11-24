import time

import argparse
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config import load_settings
from context_builder import LessonContext
from llm_solver import BaseSolver, FatalLLMError, create_solver
from step_handlers import (
    SUPPORTED_SOLVABLE_TYPES,
    THEORY_TYPES,
    build_reply,
    prepare_task,
)
from stepik_client import StepikAPIError, StepikClient

console = Console()


@dataclass
class SkippedStep:
    section_title: str
    lesson_title: str
    step_position: int
    step_type: str
    url: str
    reason: str


def save_skipped_report(course_title: str, skipped: list[SkippedStep]) -> Path | None:
    if not skipped:
        return None

    path = Path("skipped_steps.md")
    lines = [
        f"# Пропущенные шаги — {course_title}",
        "",
        "| Раздел | Урок | Шаг | Тип | Причина | Ссылка |",
        "| :--- | :--- | :---: | :--- | :--- | :--- |",
    ]
    for item in skipped:
        reason = item.reason.replace("\n", " ").replace("|", "/")[:180]
        lines.append(
            f"| {item.section_title} | {item.lesson_title} | {item.step_position} "
            f"| `{item.step_type}` | {reason} | [Открыть]({item.url}) |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def solve_interactive_step(
    client: StepikClient,
    solver: BaseSolver,
    step: dict,
    lesson_context: LessonContext,
    max_attempts: int,
) -> tuple[bool, str]:
    """Решает один интерактивный шаг с циклом самоисправления."""
    console.print("      [dim]Stepik: создаю попытку...[/dim]")
    attempt = client.create_attempt(step["id"])
    task = prepare_task(step, attempt)
    convo = solver.start_conversation(task, lesson_context.render())

    last_reason = "Неизвестная ошибка"

    for try_num in range(1, max_attempts + 1):
        console.print(f"      [dim]Попытка {try_num}/{max_attempts}: запрос к LLM...[/dim]")
        try:
            answer = convo.ask()
            reply = build_reply(task, answer)
        except FatalLLMError:
            raise
        except Exception as exc:
            last_reason = f"Ошибка генерации/сборки ответа: {exc}"
            console.print(f"    [yellow]Попытка {try_num}/{max_attempts}:[/yellow] {last_reason}")
            convo.add_feedback("validation_error", str(exc))
            continue

        if try_num > 1 and task.step_type != "code":
            attempt = client.create_attempt(step["id"])
            task = prepare_task(step, attempt)

        console.print("      [dim]Stepik: отправляю решение в грейдер...[/dim]")
        t0 = time.perf_counter()
        submission = client.submit_and_wait(attempt["id"], reply)
        grader_sec = time.perf_counter() - t0

        status = submission.get("status", "unknown")
        hint = (submission.get("hint") or "").strip()

        if status == "correct":
            console.print(f"      [dim]Грейдер подтвердил за {grader_sec:.1f}с[/dim]")
            return True, "correct"

        short_hint = hint.splitlines()[0][:120] if hint else "без подсказки"
        console.print(
            f"    [yellow]Попытка {try_num}/{max_attempts} -> {status} ({grader_sec:.1f}с)[/yellow]: {short_hint}"
        )
        last_reason = f"{status}: {hint or 'без подсказки'}"

        if try_num < max_attempts:
            convo.add_feedback(status, hint)
            client.polite_delay()

    return False, last_reason


def run(limit: int | None = None) -> None:
    settings = load_settings()
    solver = create_solver(settings)

    solved_count = 0
    viewed_count = 0
    skipped: list[SkippedStep] = []
    processed_unsolved = 0

    with StepikClient(settings) as client:
        user = client.get_current_user()
        tree = client.get_course_tree(settings.course_id)

        console.print(
            Panel.fit(
                f"[bold cyan]{tree['title']}[/bold cyan]\n"
                f"Аккаунт: [green]{user.get('full_name')}[/green] | "
                f"Модель: [magenta]{settings.llm_provider}:{settings.active_model}[/magenta]",
                title="Stepik Agent",
            )
        )

        stop_early = False

        for section in tree["sections"]:
            if stop_early:
                break
            console.print(f"\n[bold blue]== Раздел {section['position']}: {section['title']} ==[/bold blue]")

            for lesson in section["lessons"]:
                if stop_early:
                    break
                lesson_ctx = LessonContext(lesson["title"])
                has_unsolved = any(not s["is_passed"] for s in lesson["steps"])
                if not has_unsolved:
                    continue

                console.print(f"  [bold]Урок:[/bold] {lesson['title']}")

                for step in lesson["steps"]:
                    block = step.get("block", {})
                    step_type = block.get("name", "unknown")

                    # Теорию всегда добавляем в контекст урока (даже если шаг уже пройден)
                    if step_type in THEORY_TYPES:
                        lesson_ctx.add_theory(step["position"], block.get("text"))

                    if step["is_passed"]:
                        continue

                    if limit is not None and processed_unsolved >= limit:
                        stop_early = True
                        break

                    processed_unsolved += 1
                    prefix = f"    Шаг {step['position']} [{step_type}]"

                    if step_type in THEORY_TYPES:
                        try:
                            client.mark_step_viewed(step["id"], step["assignment_id"])
                            viewed_count += 1
                            console.print(f"{prefix} -> [green]просмотрен[/green]")
                        except StepikAPIError as exc:
                            console.print(f"{prefix} -> [red]ошибка просмотра: {exc}[/red]")
                            skipped.append(
                                SkippedStep(
                                    section["title"],
                                    lesson["title"],
                                    step["position"],
                                    step_type,
                                    step["url"],
                                    str(exc),
                                )
                            )
                        client.polite_delay()
                        continue

                    if step_type not in SUPPORTED_SOLVABLE_TYPES:
                        console.print(f"{prefix} -> [dim]пропущен (неподдерживаемый тип)[/dim]")
                        skipped.append(
                            SkippedStep(
                                section["title"],
                                lesson["title"],
                                step["position"],
                                step_type,
                                step["url"],
                                f"Тип {step_type} требует ручного решения",
                            )
                        )
                        continue

                    console.print(f"{prefix} -> [cyan]решаю...[/cyan]")
                    try:
                        ok, reason = solve_interactive_step(
                            client=client,
                            solver=solver,
                            step=step,
                            lesson_context=lesson_ctx,
                            max_attempts=settings.max_attempts_per_step,
                        )
                        if ok:
                            solved_count += 1
                            console.print(f"{prefix} -> [bold green]РЕШЕНО[/bold green]")
                        else:
                            console.print(f"{prefix} -> [bold red]НЕ РЕШЕНО[/bold red]")
                            skipped.append(
                                SkippedStep(
                                    section["title"],
                                    lesson["title"],
                                    step["position"],
                                    step_type,
                                    step["url"],
                                    reason,
                                )
                            )
                    except Exception as exc:
                        console.print(f"{prefix} -> [red]сбой: {exc}[/red]")
                        skipped.append(
                            SkippedStep(
                                section["title"],
                                lesson["title"],
                                step["position"],
                                step_type,
                                step["url"],
                                str(exc),
                            )
                        )

                    client.polite_delay()

    report_path = save_skipped_report(tree["title"], skipped)
    console.print("\n[bold]Итог запуска:[/bold]")
    console.print(f"  Решено задач: [green]{solved_count}[/green]")
    console.print(f"  Отмечено теории: [cyan]{viewed_count}[/cyan]")
    console.print(f"  Пропущено / не решено: [yellow]{len(skipped)}[/yellow]")
    if report_path:
        console.print(f"  Отчет сохранен в: [bold]{report_path.resolve()}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автоматическое прохождение курсов Stepik.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество обрабатываемых нерешенных шагов за один запуск.",
    )
    args = parser.parse_args()
    run(limit=args.limit)