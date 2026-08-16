"""Tests for examples/gsm8k_distillation/novelty_search/embedding.py.

The module imports `langchain_aws` at import time, which is an examples-only dependency; when it
is not installed a stub is registered so the pure embedding math (normalization, one-hot
difficulty encoding, block concatenation) can still be tested without AWS or network access.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _import_embedding_module():
    if "langchain_aws" not in sys.modules:
        try:
            importlib.import_module("langchain_aws")
        except ImportError:
            stub = types.ModuleType("langchain_aws")

            class BedrockEmbeddings:  # pragma: no cover - only records constructor args
                def __init__(self, model_id):
                    self.model_id = model_id

            stub.BedrockEmbeddings = BedrockEmbeddings
            sys.modules["langchain_aws"] = stub
    return importlib.import_module("examples.gsm8k_distillation.novelty_search.embedding")


embedding = _import_embedding_module()


class FakeClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return self.vectors


@pytest.fixture
def reset_client_cache():
    embedding._embeddings_client = None
    yield
    embedding._embeddings_client = None


class TestEmbeddingsClient:
    def test_client_is_created_once_and_cached(self, reset_client_cache):
        first = embedding._get_embeddings_client(embedding.DEFAULT_MODEL_ID)
        second = embedding._get_embeddings_client("some-other-model")
        assert first is second
        assert first.model_id == embedding.DEFAULT_MODEL_ID


class TestQuestionEmbeddingFn:
    def test_rows_are_unit_norm(self, reset_client_cache, monkeypatch):
        client = FakeClient([[3.0, 4.0], [0.0, 2.0]])
        monkeypatch.setattr(embedding, "_get_embeddings_client", lambda model_id: client)

        vectors = embedding.question_embedding_fn(["a", "b"])
        assert vectors.shape == (2, 2)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(vectors[0], [0.6, 0.8])

    def test_zero_vector_is_left_untouched(self, reset_client_cache, monkeypatch):
        client = FakeClient([[0.0, 0.0]])
        monkeypatch.setattr(embedding, "_get_embeddings_client", lambda model_id: client)

        vectors = embedding.question_embedding_fn(["a"])
        np.testing.assert_allclose(vectors[0], [0.0, 0.0])

    def test_questions_are_forwarded_as_one_batch(self, reset_client_cache, monkeypatch):
        client = FakeClient([[1.0], [1.0], [1.0]])
        monkeypatch.setattr(embedding, "_get_embeddings_client", lambda model_id: client)

        embedding.question_embedding_fn(["a", "b", "c"])
        assert client.calls == [["a", "b", "c"]]


class TestDifficultyOneHot:
    def test_shape_is_num_samples_by_k_plus_one(self):
        one_hot = embedding.difficulty_one_hot([0, 2, 4], k=4)
        assert one_hot.shape == (3, 5)

    def test_sets_exactly_the_labelled_index(self):
        one_hot = embedding.difficulty_one_hot([0, 3], k=3)
        assert one_hot[0].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert one_hot[1].tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_rows_are_unit_norm(self):
        one_hot = embedding.difficulty_one_hot([0, 1, 2], k=2)
        np.testing.assert_allclose(np.linalg.norm(one_hot, axis=1), np.ones(3))

    def test_empty_input(self):
        assert embedding.difficulty_one_hot([], k=3).shape == (0, 4)

    def test_out_of_range_label_raises(self):
        with pytest.raises(IndexError):
            embedding.difficulty_one_hot([5], k=2)


class TestCombineEmbeddings:
    def test_concatenates_question_completion_and_difficulty_blocks(self):
        sem = np.array([[1.0, 0.0]])
        completion = np.array([[0.0, 1.0]])
        combined = embedding.combine_embeddings(sem, completion, [1], k=2)

        assert len(combined) == 1
        assert combined[0].shape == (2 + 2 + 3,)
        np.testing.assert_allclose(combined[0], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    def test_normalizes_each_semantic_block_independently(self):
        sem = np.array([[3.0, 4.0]])
        completion = np.array([[0.0, 5.0]])
        combined = embedding.combine_embeddings(sem, completion, [0], k=1)

        np.testing.assert_allclose(combined[0][:2], [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(combined[0][2:4], [0.0, 1.0], atol=1e-6)

    def test_each_block_contributes_equal_norm(self):
        sem = np.array([[10.0, 0.0]])
        completion = np.array([[0.0, 0.5]])
        combined = embedding.combine_embeddings(sem, completion, [2], k=2)

        blocks = [combined[0][:2], combined[0][2:4], combined[0][4:]]
        np.testing.assert_allclose([np.linalg.norm(b) for b in blocks], [1.0, 1.0, 1.0], atol=1e-6)

    def test_zero_blocks_are_not_normalized(self):
        sem = np.array([[0.0, 0.0]])
        completion = np.array([[0.0, 0.0]])
        combined = embedding.combine_embeddings(sem, completion, [0], k=1)
        np.testing.assert_allclose(combined[0][:4], np.zeros(4))

    def test_returns_float32_vectors(self):
        sem = np.array([[1.0, 0.0], [0.0, 1.0]])
        combined = embedding.combine_embeddings(sem, sem, [0, 1], k=1)
        assert all(v.dtype == np.float32 for v in combined)

    def test_one_vector_per_difficulty_label(self):
        sem = np.ones((3, 2))
        combined = embedding.combine_embeddings(sem, sem, [0, 1, 2], k=2)
        assert len(combined) == 3

    def test_empty_batch(self):
        combined = embedding.combine_embeddings(np.empty((0, 2)), np.empty((0, 2)), [], k=1)
        assert combined == []
