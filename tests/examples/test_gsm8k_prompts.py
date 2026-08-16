"""Tests for examples/gsm8k_distillation/novelty_search/prompts.py.

Prompt builders and the answer parsing/comparison helpers are pure and dependency-free, so they
are tested directly (the LLM-backed callers around them are not).
"""

from __future__ import annotations

import pytest

from examples.gsm8k_distillation.novelty_search import prompts


class TestTopics:
    def test_topics_are_unique_non_empty_strings(self):
        assert prompts.TOPICS
        assert len(set(prompts.TOPICS)) == len(prompts.TOPICS)
        assert all(isinstance(t, str) and t.strip() for t in prompts.TOPICS)


class TestBuildMutationPrompt:
    def test_includes_parent_question_and_topic(self):
        prompt = prompts.build_mutation_prompt("Jo has 3 apples.", "shopping and money")
        assert "Jo has 3 apples." in prompt
        assert "context of shopping and money" in prompt

    def test_asks_for_exactly_one_bare_question(self):
        prompt = prompts.build_mutation_prompt("q", "t")
        assert "ONE new grade-school arithmetic word problem" in prompt
        assert "Respond with ONLY the new question text" in prompt

    def test_handles_multiline_parent_question(self):
        parent = "Line one.\nLine two."
        assert parent in prompts.build_mutation_prompt(parent, "t")


class TestBuildCrossoverPrompt:
    def test_includes_both_parents_and_topic(self):
        prompt = prompts.build_crossover_prompt("first q", "second q", "sports and games")
        assert "Example question 1:\nfirst q" in prompt
        assert "Example question 2:\nsecond q" in prompt
        assert "context of sports and games" in prompt

    def test_asks_to_combine_both_techniques(self):
        prompt = prompts.build_crossover_prompt("a", "b", "t")
        assert "combines elements of both" in prompt
        assert "Respond with ONLY the new question text" in prompt

    def test_identical_parents_appear_in_both_slots(self):
        prompt = prompts.build_crossover_prompt("same", "same", "t")
        assert prompt.count("same") == 2


class TestBuildBaselineGenerationPrompt:
    def test_renders_seed_questions_as_bullets(self):
        prompt = prompts.build_baseline_generation_prompt(
            ["q one", "q two"], "school supplies", "easy", "2 steps"
        )
        assert "- q one\n- q two" in prompt

    def test_includes_topic_difficulty_and_step_budget(self):
        prompt = prompts.build_baseline_generation_prompt(["q"], "travel", "hard", "3 steps")
        assert "about travel" in prompt
        assert "at a hard difficulty level" in prompt
        assert "roughly 3 steps" in prompt

    def test_empty_seed_list_still_builds_a_prompt(self):
        prompt = prompts.build_baseline_generation_prompt([], "travel", "easy", "1 step")
        assert "Respond with ONLY the new question text" in prompt


class TestBuildSolvePrompt:
    def test_includes_question_and_answer_format_contract(self):
        prompt = prompts.build_solve_prompt("How many apples?")
        assert "Question: How many apples?" in prompt
        assert "'#### <number>'" in prompt
        assert prompt.rstrip().endswith("Answer:")


class TestExtractQuestion:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  A question?  ", "A question?"),
            ('"A question?"', "A question?"),
            ("'A question?'", "A question?"),
            ("`A question?`", "A question?"),
            ("```A question?```", "A question?"),
            ("Question: A question?", "A question?"),
            ("question : A question?", "A question?"),
            ("NEW QUESTION: A question?", "A question?"),
            ("A question?", "A question?"),
        ],
    )
    def test_strips_wrappers_and_labels(self, raw, expected):
        assert prompts.extract_question(raw) == expected

    def test_preserves_inner_quotes(self):
        assert prompts.extract_question('Ann said "hi" to Bob.') == 'Ann said "hi" to Bob.'

    def test_preserves_internal_newlines(self):
        assert prompts.extract_question("  line one\nline two  ") == "line one\nline two"

    def test_empty_input(self):
        assert prompts.extract_question("   ") == ""


class TestExtractGsm8kAnswer:
    @pytest.mark.parametrize(
        "completion, expected",
        [
            ("reasoning...\n#### 42", "42"),
            ("#### -7", "-7"),
            ("#### 1,234", "1,234"),
            ("#### 3.5", "3.5"),
            ("####42", "42"),
            ("####    42", "42"),
        ],
    )
    def test_parses_terminator_line(self, completion, expected):
        assert prompts.extract_gsm8k_answer(completion) == expected

    @pytest.mark.parametrize("completion", ["no terminator here", "", "#### abc"])
    def test_returns_none_without_a_numeric_answer(self, completion):
        assert prompts.extract_gsm8k_answer(completion) is None

    def test_returns_last_match_when_multiple(self):
        # the model's own final answer, not a marker quoted from the prompt's few-shot examples
        assert prompts.extract_gsm8k_answer("#### 1\nmore\n#### 2") == "2"


class TestAnswersMatch:
    @pytest.mark.parametrize(
        "predicted, true",
        [
            ("42", "42"),
            ("42", "42.0"),
            ("1,234", "1234"),
            ("-7", "-7"),
            ("  42  ", "42"),
        ],
    )
    def test_numerically_equal_answers_match(self, predicted, true):
        assert prompts.answers_match(predicted, true) is True

    @pytest.mark.parametrize(
        "predicted, true",
        [
            ("42", "43"),
            ("42", None),
            (None, "42"),
            (None, None),
            ("abc", "42"),
            ("42", "abc"),
            ("", "42"),
        ],
    )
    def test_mismatched_or_unparseable_answers_do_not_match(self, predicted, true):
        assert prompts.answers_match(predicted, true) is False

    def test_round_trip_with_extract_gsm8k_answer(self):
        completion = "so the answer is\n#### 1,024"
        ground_truth = "steps...\n#### 1024"
        assert prompts.answers_match(
            prompts.extract_gsm8k_answer(completion),
            prompts.extract_gsm8k_answer(ground_truth),
        )
