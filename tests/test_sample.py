"""Tests for EvoSample."""

from __future__ import annotations

import numpy as np
import pytest

from novelty_search_evolution.sample import EvoSample, Status


class TestStatusTransitions:
    def test_default_status_is_stale(self):
        s = EvoSample(data="x")
        assert s.get_status() == Status.STALE

    def test_set_active(self):
        s = EvoSample(data="x")
        s.set_active()
        assert s.get_status() == Status.ACTIVE

    def test_set_inactive(self):
        s = EvoSample(data="x")
        s.set_inactive()
        assert s.get_status() == Status.INACTIVE

    def test_set_reject(self):
        s = EvoSample(data="x")
        s.set_reject()
        assert s.get_status() == Status.REJECTED

    def test_set_stale(self):
        s = EvoSample(data="x")
        s.set_active()
        s.set_stale()
        assert s.get_status() == Status.STALE

    @pytest.mark.parametrize(
        "method", ["set_active", "set_stale", "set_inactive", "set_reject"]
    )
    def test_feedback_appended_on_transition(self, method):
        s = EvoSample(data="x")
        getattr(s, method)(feedback="note")
        assert s.get_feedback() == ["note"]

    def test_no_feedback_when_none(self):
        s = EvoSample(data="x")
        s.set_active(feedback=None)
        assert s.get_feedback() == []

    def test_feedback_accumulates_across_transitions(self):
        s = EvoSample(data="x")
        s.set_active(feedback="a")
        s.set_inactive(feedback="b")
        assert s.get_feedback() == ["a", "b"]


class TestEmbedding:
    def test_get_embedding_raises_before_set(self):
        s = EvoSample(data="x")
        with pytest.raises(ValueError):
            s.get_embedding()

    def test_set_and_get_embedding(self):
        s = EvoSample(data="x")
        emb = np.array([1.0, 2.0])
        s._set_embedding(emb)
        assert np.array_equal(s.get_embedding(), emb)

    def test_set_embedding_twice_raises(self):
        s = EvoSample(data="x")
        s._set_embedding(np.array([1.0]))
        with pytest.raises(ValueError):
            s._set_embedding(np.array([2.0]))


class TestGenealogy:
    def test_add_child(self):
        parent = EvoSample(data="p")
        child = EvoSample(data="c", parents=[parent])
        parent._add_child(child)
        assert parent.get_children() == [child]

    def test_sample_accepted_children_filters_rejected(self):
        parent = EvoSample(data="p")
        accepted_child = EvoSample(data="a")
        accepted_child.set_stale()
        rejected_child = EvoSample(data="r")
        rejected_child.set_reject()
        parent._add_child(accepted_child)
        parent._add_child(rejected_child)

        assert parent.sample_accepted_children() == [accepted_child]
        assert parent.sample_rejected_children() == [rejected_child]

    def test_sample_accepted_children_respects_k(self):
        parent = EvoSample(data="p")
        children = []
        for i in range(5):
            c = EvoSample(data=i)
            c.set_stale()
            parent._add_child(c)
            children.append(c)

        result = parent.sample_accepted_children(k=2)
        assert len(result) == 2
        assert all(c in children for c in result)

    def test_sample_accepted_children_default_all(self):
        parent = EvoSample(data="p")
        for i in range(3):
            c = EvoSample(data=i)
            c.set_stale()
            parent._add_child(c)
        assert len(parent.sample_accepted_children()) == 3


class TestDepth:
    def test_depth_zero_for_parentless_sample(self):
        s = EvoSample(data="root")
        assert s.get_depth() == 0

    def test_depth_one_for_single_parent_child(self):
        parent = EvoSample(data="p")
        child = EvoSample(data="c", parents=[parent])
        assert child.get_depth() == 1

    def test_depth_is_max_of_parents_plus_one(self):
        root = EvoSample(data="root")
        mid = EvoSample(data="mid", parents=[root])
        deep = EvoSample(data="deep", parents=[mid])
        assert deep.get_depth() == 2

        child = EvoSample(data="child", parents=[root, deep])
        assert child.get_depth() == 3


class TestEquality:
    def test_equality_by_id(self):
        s1 = EvoSample(data="x", id="abc")
        s2 = EvoSample(data="y", id="abc")
        assert s1 == s2

    def test_inequality_different_ids(self):
        s1 = EvoSample(data="x")
        s2 = EvoSample(data="x")
        assert s1 != s2

    def test_hash_by_id(self):
        s1 = EvoSample(data="x", id="abc")
        s2 = EvoSample(data="y", id="abc")
        assert {s1, s2} == {s1}
