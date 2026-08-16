"""Tests for render_population_html()."""

from __future__ import annotations

from novelty_search_evolution.population_viewer import render_population_html


def make_samples():
    return [
        {
            "id": "root-1",
            "data": {"prompt": "improve this solution", "solution": "def train(): ...", "accuracy": 0.8},
            "status": "active",
            "generation": 0,
            "depth": 0,
            "parent_ids": [],
            "child_ids": ["child-1"],
            "feedback": [],
        },
        {
            "id": "child-1",
            "data": {"prompt": "mutate further", "solution": "def train(): ...v2", "accuracy": 0.85},
            "status": "stale",
            "generation": 1,
            "depth": 1,
            "parent_ids": ["root-1"],
            "child_ids": [],
            "feedback": ["looked promising"],
        },
    ]


class TestRenderPopulationHtml:
    def test_returns_well_formed_html_document(self):
        html = render_population_html(make_samples())
        assert html.strip().startswith("<!doctype html>")
        assert "</html>" in html

    def test_embeds_sample_ids(self):
        html = render_population_html(make_samples())
        assert "root-1" in html
        assert "child-1" in html

    def test_embeds_prompt_and_solution_values(self):
        html = render_population_html(make_samples())
        assert "improve this solution" in html
        assert "mutate further" in html
        assert "def train(): ...v2" in html

    def test_embeds_parent_and_child_links(self):
        html = render_population_html(make_samples())
        # parent_ids/child_ids values appear in the embedded JSON
        assert '"parent_ids": ["root-1"]' in html
        assert '"child_ids": ["child-1"]' in html

    def test_no_external_resources(self):
        html = render_population_html(make_samples())
        assert "<script src=" not in html
        assert "<link href=" not in html
        assert "http://" not in html
        assert "https://" not in html

    def test_handles_empty_sample_list(self):
        html = render_population_html([])
        assert html.strip().startswith("<!doctype html>")
        assert "const SAMPLES = []" in html

    def test_embedded_json_cannot_break_out_of_script_block(self):
        samples = make_samples()
        samples[0]["data"] = {"prompt": "</script><img src=x onerror=alert(1)>", "solution": "x"}
        samples[0]["feedback"] = ["<!--<script>"]
        html = render_population_html(samples)
        assert "</script><img" not in html
        assert html.count("</script>") == 1
        assert "\\u003c/script\\u003e" in html

    def test_handles_non_dict_data(self):
        samples = [
            {
                "id": "s1",
                "data": (0.5, 0.5),
                "status": "active",
                "generation": 0,
                "depth": 0,
                "parent_ids": [],
                "child_ids": [],
                "feedback": [],
            }
        ]
        html = render_population_html(samples)
        assert "0.5" in html

    def test_escapes_script_terminator_in_data(self):
        samples = [
            {
                "id": "s1",
                "data": "</script><h1>pwned</h1>",
                "status": "active",
                "generation": 0,
                "depth": 0,
                "parent_ids": [],
                "child_ids": [],
                "feedback": [],
            }
        ]
        html = render_population_html(samples)
        assert "</script><h1>" not in html
        assert "\\u003c/script\\u003e" in html
