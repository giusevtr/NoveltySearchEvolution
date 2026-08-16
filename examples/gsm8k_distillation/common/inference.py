"""Teacher/student sampling steps shared by generation, baseline and eval scripts.

Every arm must elicit completions with byte-identical prompts and identical inference
parameters for the comparison between them to be meaningful, so both live here rather than
being restated per script.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.gsm8k_distillation.novelty_search.prompts import (
    answers_match,
    build_solve_prompt,
    extract_gsm8k_answer,
    extract_question,
)

if TYPE_CHECKING:
    from examples.gsm8k_distillation.clients.student_client import StudentClient
    from examples.gsm8k_distillation.clients.teacher_client import TeacherClient

K_STUDENT_SAMPLES = 10  # difficulty label range: [0, K_STUDENT_SAMPLES]

MUTATION_TEMPERATURE = 0.9
TEACHER_SOLVE_TEMPERATURE = 0.7
STUDENT_SOLVE_TEMPERATURE = 0.8
MUTATION_MAX_TOKENS = 256
SOLVE_MAX_TOKENS = 512


def teacher_write_questions(teacher: TeacherClient, prompts: list[str]) -> list[str]:
    """Run mutation/crossover prompts through the teacher and strip each answer to bare question text."""
    raw_questions = teacher.generate(prompts, max_tokens=MUTATION_MAX_TOKENS, temperature=MUTATION_TEMPERATURE)
    return [extract_question(q) for q in raw_questions]


def teacher_solve(
    teacher: TeacherClient, questions: list[str], max_tokens: int = SOLVE_MAX_TOKENS
) -> list[str]:
    """One teacher completion per question, treated as that question's ground truth."""
    return teacher.generate(
        [build_solve_prompt(q) for q in questions], max_tokens=max_tokens, temperature=TEACHER_SOLVE_TEMPERATURE
    )


def student_solve(
    student: StudentClient, questions: list[str], n: int = K_STUDENT_SAMPLES
) -> list[list[str]]:
    """`n` student completions per question."""
    return student.generate(
        [build_solve_prompt(q) for q in questions],
        n=n,
        temperature=STUDENT_SOLVE_TEMPERATURE,
        max_tokens=SOLVE_MAX_TOKENS,
    )


def score_difficulty(
    student: StudentClient, questions: list[str], teacher_answers: list[str | None], k: int = K_STUDENT_SAMPLES
) -> list[int]:
    """For each question, sample k student completions and count matches vs the teacher answer."""
    student_completions = student_solve(student, questions, n=k)
    return [
        sum(1 for c in completions if answers_match(extract_gsm8k_answer(c), teacher_answer))
        for completions, teacher_answer in zip(student_completions, teacher_answers)
    ]


def solve_and_score(
    teacher: TeacherClient, student: StudentClient, questions: list[str]
) -> tuple[list[str], list[int]]:
    """Teacher-solve each question, then difficulty-score it against that teacher answer."""
    teacher_completions = teacher_solve(teacher, questions)
    teacher_answers = [extract_gsm8k_answer(c) for c in teacher_completions]
    return teacher_completions, score_difficulty(student, questions, teacher_answers)
