"""Prompt templates and parsing helpers for GSM8K data augmentation."""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"####\s*(-?[\d,\.]+)")

# Shared topic diversity factor: used both as novelty search's mutation-prompt conditioning
# (build_mutation_prompt) and as baseline_nemo.py's Data Designer sampler column, so both arms
# draw from the same topic pool for a fair comparison.
TOPICS = [
    "shopping and money",
    "cooking and recipes",
    "sports and games",
    "travel and distance",
    "school supplies",
    "time and schedules",
    "measurement and construction",
    "saving and allowance",
]


def build_mutation_prompt(parent_question: str, topic: str) -> str:
    return (
        "You are creating training data for a grade-school math tutoring system.\n"
        "Given the example question below, write ONE new grade-school arithmetic "
        "word problem that is a variant of it: paraphrase the scenario, change the "
        "entities/names, change the numbers, and optionally add or remove a "
        "reasoning step to make it easier or harder. Set the new question in the "
        f"context of {topic}. The new question must be "
        "self-contained and solvable using only basic arithmetic (+, -, *, /).\n\n"
        f"Example question:\n{parent_question}\n\n"
        "Respond with ONLY the new question text, no preamble, no explanation, "
        "no numbering."
    )


def build_crossover_prompt(parent_question_1: str, parent_question_2: str, topic: str) -> str:
    return (
        "You are creating training data for a grade-school math tutoring system.\n"
        "Below are two example grade-school arithmetic word problems. Identify the "
        "arithmetic technique or reasoning structure each one relies on (for example: "
        "multi-step subtraction, rate x time, unit conversion, percentage of a total). "
        "Then write ONE new grade-school word problem that combines elements of both "
        f"techniques into a single problem. Set the new question in the context of {topic}. "
        "The new question must be self-contained and solvable using only basic arithmetic "
        "(+, -, *, /).\n\n"
        f"Example question 1:\n{parent_question_1}\n\n"
        f"Example question 2:\n{parent_question_2}\n\n"
        "Respond with ONLY the new question text, no preamble, no explanation, "
        "no numbering."
    )


def build_baseline_generation_prompt(seed_questions: list[str], topic: str, difficulty: str, num_steps: str) -> str:
    examples = "\n".join(f"- {q}" for q in seed_questions)
    return (
        "You are creating training data for a grade-school math tutoring system.\n"
        "Here are some example grade-school arithmetic word problems:\n\n"
        f"{examples}\n\n"
        "Write ONE new grade-school arithmetic word problem in a similar style to the examples "
        f"above, about {topic}, at a {difficulty} difficulty level, that takes roughly {num_steps} "
        "to solve. The new question must be self-contained, distinct from the examples above, and "
        "solvable using only basic arithmetic (+, -, *, /).\n\n"
        "Respond with ONLY the new question text, no preamble, no explanation, no numbering."
    )


def build_solve_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math word problem. Show your step-by-step "
        "reasoning, then on the final line write the answer in exactly this format: "
        "'#### <number>' (just the final numeric answer, no units, no commas).\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def extract_question(raw_text: str) -> str:
    """Strip any markdown/preamble a model might add around the question text."""
    text = raw_text.strip()
    text = re.sub(r'^["\'`]+|["\'`]+$', "", text)
    text = re.sub(r"^(question|new question)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_gsm8k_answer(completion: str) -> str | None:
    """Parse the final numeric answer after the '#### ' terminator, or None if missing."""
    match = _ANSWER_RE.search(completion)
    return match.group(1) if match else None


def answers_match(predicted: str | None, true: str | None) -> bool:
    if predicted is None or true is None:
        return False
    try:
        return float(predicted.replace(",", "")) == float(true.replace(",", ""))
    except ValueError:
        return False
