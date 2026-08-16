"""Tests for Population."""

from __future__ import annotations

import numpy as np
import pytest

from novelty_search_evolution.population import Population
from novelty_search_evolution.sample import EvoSample, Status


def make_embedding_fn(calls):
    def embedding_fn(data_list):
        calls.append(len(data_list))
        return [np.array([float(d)]) for d in data_list]

    return embedding_fn


class TestSeeds:
    def test_set_seeds_marks_active(self):
        pop = Population()
        pop.set_seeds([1, 2, 3])
        assert len(pop.get_active()) == 3

    def test_set_seeds_archives_unconditionally(self):
        pop = Population()
        pop.set_seeds([1, 2, 3])
        assert len(pop.get_archive()) == 3

    def test_seeds_generation_zero(self):
        pop = Population()
        pop.set_seeds([1])
        assert pop.get_all_samples()[0].get_generation() == 0


class TestArchive:
    def test_archive_preserves_insertion_order(self):
        pop = Population()
        a = EvoSample(data="a")
        b = EvoSample(data="b")
        pop.add(a)
        pop.add(b)
        pop.add_to_archive(b)
        pop.add_to_archive(a)
        assert [s.get_id() for s in pop.get_archive()] == [b.get_id(), a.get_id()]

    def test_add_to_archive_idempotent(self):
        pop = Population()
        a = EvoSample(data="a")
        pop.add(a)
        pop.add_to_archive(a)
        pop.add_to_archive(a)
        assert len(pop.get_archive()) == 1


class TestStatusGetters:
    def test_status_getters_reflect_manual_transitions(self):
        pop = Population()
        a, b, c, d = (EvoSample(data=i) for i in range(4))
        pop.bulk_add([a, b, c, d])
        a.set_active()
        b.set_inactive()
        c.set_reject()
        # d stays stale

        assert pop.get_active() == [a]
        assert pop.get_inactive() == [b]
        assert pop.get_rejected() == [c]
        assert pop.get_stale() == [d]
        assert pop.get_by_status(Status.ACTIVE) == [a]

    def test_get_by_id(self):
        pop = Population()
        a = EvoSample(data="a")
        pop.add(a)
        assert pop.get_by_id(a.get_id()) is a

    def test_size_and_size_by_status(self):
        pop = Population()
        a, b = EvoSample(data=1), EvoSample(data=2)
        pop.bulk_add([a, b])
        a.set_active()
        assert pop.size() == 2
        counts = pop.size_by_status()
        assert counts[Status.ACTIVE] == 1
        assert counts[Status.STALE] == 1


class TestEmbeddings:
    def test_compute_embeddings_calls_fn_once_per_batch(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        samples = [EvoSample(data=i) for i in range(3)]
        pop.bulk_add(samples)

        pop.compute_embeddings(samples)
        assert calls == [3]
        for s in samples:
            assert s.get_embedding() is not None

    def test_compute_embeddings_skips_already_embedded(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        samples = [EvoSample(data=i) for i in range(3)]
        pop.bulk_add(samples)

        pop.compute_embeddings(samples)
        pop.compute_embeddings(samples)
        assert calls == [3]  # second call is a no-op

    def test_compute_embeddings_only_new_samples(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        s1 = EvoSample(data=1)
        pop.add(s1)
        pop.compute_embeddings([s1])

        s2 = EvoSample(data=2)
        pop.add(s2)
        pop.compute_embeddings([s1, s2])
        assert calls == [1, 1]  # second batch only embeds s2

    def test_get_embeddings_preserves_order(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        samples = [EvoSample(data=i) for i in [5, 1, 3]]
        pop.bulk_add(samples)
        embeddings = pop.get_embeddings(samples)
        assert embeddings.tolist() == [[5.0], [1.0], [3.0]]

    def test_get_archive_embeddings(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        pop.set_seeds([1, 2])
        embeddings = pop.get_archive_embeddings()
        assert embeddings.shape == (2, 1)

    def test_compute_embeddings_without_embedding_fn_raises(self):
        pop = Population()
        s = EvoSample(data=1)
        pop.add(s)
        with pytest.raises(ValueError, match="No embedding_fn set"):
            pop.compute_embeddings([s])

    def test_compute_embeddings_without_embedding_fn_is_noop_when_nothing_missing(self):
        pop = Population()
        s = EvoSample(data=1)
        s._set_embedding(np.array([1.0]))
        pop.add(s)
        pop.compute_embeddings([s])  # no embedding_fn needed — nothing to compute

    def test_compute_embeddings_defaults_to_whole_population(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        pop.bulk_add([EvoSample(data=i) for i in range(4)])
        pop.compute_embeddings()
        assert calls == [4]

    def test_get_embeddings_of_empty_list_returns_empty_matrix(self):
        pop = Population()
        assert pop.get_embeddings([]).shape == (0, 0)

    def test_get_archive_embeddings_empty_archive(self):
        pop = Population()
        assert pop.get_archive_embeddings().size == 0


class TestEmbeddingMatrix:
    def test_matrix_defaults_to_all_samples(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        samples = [EvoSample(data=i) for i in [4, 5]]
        pop.bulk_add(samples)

        matrix, returned = pop.get_embedding_matrix()
        assert matrix.tolist() == [[4.0], [5.0]]
        assert returned == samples

    def test_matrix_filtered_by_status(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        active, stale = EvoSample(data=7), EvoSample(data=8)
        pop.bulk_add([active, stale])
        active.set_active()

        matrix, returned = pop.get_embedding_matrix(status=Status.ACTIVE)
        assert matrix.tolist() == [[7.0]]
        assert returned == [active]

    def test_matrix_rows_align_with_returned_samples(self):
        pop = Population()
        calls = []
        pop.set_embedding_column(make_embedding_fn(calls))
        pop.bulk_add([EvoSample(data=i) for i in [3, 1, 2]])

        matrix, returned = pop.get_embedding_matrix()
        for row, sample in zip(matrix, returned):
            assert row.tolist() == [float(sample.get_data())]


class TestSampleRandom:
    def test_sample_random_with_status(self):
        pop = Population()
        samples = [EvoSample(data=i) for i in range(5)]
        pop.bulk_add(samples)
        for s in samples:
            s.set_active()
        result = pop.sample_random(3, status=Status.ACTIVE)
        assert len(result) == 3

    def test_sample_random_more_than_pool_returns_all(self):
        pop = Population()
        samples = [EvoSample(data=i) for i in range(2)]
        pop.bulk_add(samples)
        for s in samples:
            s.set_active()
        result = pop.sample_random(10, status=Status.ACTIVE)
        assert len(result) == 2
    def test_sample_random_without_status_uses_whole_population(self):
        pop = Population()
        samples = [EvoSample(data=i) for i in range(4)]
        pop.bulk_add(samples)
        samples[0].set_reject()

        result = pop.sample_random(3)
        assert len(result) == 3
        assert all(s in samples for s in result)

    def test_sample_random_returns_distinct_samples(self):
        pop = Population()
        samples = [EvoSample(data=i) for i in range(5)]
        pop.bulk_add(samples)
        for s in samples:
            s.set_active()
        result = pop.sample_random(4, status=Status.ACTIVE)
        assert len({s.get_id() for s in result}) == 4

    def test_sample_random_empty_pool_returns_empty(self):
        pop = Population()
        pop.add(EvoSample(data=1))  # stale, so no active samples to draw from
        assert pop.sample_random(3, status=Status.ACTIVE) == []

    def test_sample_random_zero_returns_empty(self):
        pop = Population()
        pop.set_seeds([1, 2])
        assert pop.sample_random(0, status=Status.ACTIVE) == []


class TestComputeEmbeddingsValidation:
    def test_embedding_fn_returning_wrong_count_raises(self):
        pop = Population()
        pop.set_embedding_column(lambda data_list: [np.array([1.0])])
        samples = [EvoSample(data=i) for i in range(3)]
        pop.bulk_add(samples)
        with pytest.raises(ValueError, match="one embedding per input"):
            pop.compute_embeddings(samples)
