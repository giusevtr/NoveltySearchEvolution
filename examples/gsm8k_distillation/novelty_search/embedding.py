"""Semantic + difficulty embedding for GSM8K novelty search.

Combines a question's semantic embedding, its teacher completion's semantic embedding (both
Bedrock Titan text embeddings via langchain-aws), and a one-hot encoding of its difficulty label
(0..K, the number of times the student model solved it out of K samples) into a single
unit-norm-per-block embedding, so all three objectives contribute equally to cosine/euclidean
distance computations. Including the completion embedding rewards novelty in *how* a question is
solved (reasoning path), not just novelty in the question text itself.
"""

from __future__ import annotations

import numpy as np
from langchain_aws import BedrockEmbeddings

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"

_embeddings_client = None


def _get_embeddings_client(model_id: str) -> BedrockEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = BedrockEmbeddings(model_id=model_id)
    return _embeddings_client


def question_embedding_fn(questions: list[str], model_id: str = DEFAULT_MODEL_ID) -> np.ndarray:
    """Compute unit-norm semantic embeddings for a batch of questions via Bedrock.

    Returns:
        (N, D) array of unit-norm rows.
    """
    client = _get_embeddings_client(model_id)
    vectors = np.asarray(client.embed_documents(questions), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def difficulty_one_hot(difficulties: list[int], k: int) -> np.ndarray:
    """One-hot encode difficulty labels in [0, k] into (N, k+1) unit-norm rows."""
    one_hot = np.zeros((len(difficulties), k + 1), dtype=np.float64)
    for i, d in enumerate(difficulties):
        if not 0 <= d <= k:
            raise ValueError(f"difficulty {d} at index {i} is outside the valid range [0, {k}]")
        one_hot[i, d] = 1.0
    return one_hot


def combine_embeddings(
    sem_vecs: np.ndarray, completion_vecs: np.ndarray, difficulties: list[int], k: int
) -> list[np.ndarray]:
    """Concatenate unit-norm question/completion semantic embeddings with a unit-norm one-hot
    difficulty block.

    Args:
        sem_vecs: (N, D) unit-norm question semantic embedding rows.
        completion_vecs: (N, D) unit-norm teacher-completion semantic embedding rows.
        difficulties: length-N list of integer difficulty labels in [0, k].
        k: max difficulty value (number of student samples per question).

    Returns:
        List of N combined embedding vectors, each
        [question_block (unit-norm), completion_block (unit-norm), difficulty_block (unit-norm)].
    """
    if len(sem_vecs) != len(difficulties) or len(completion_vecs) != len(difficulties):
        raise ValueError(
            f"row count mismatch: {len(sem_vecs)} question vectors, "
            f"{len(completion_vecs)} completion vectors, {len(difficulties)} difficulties"
        )
    diff_blocks = difficulty_one_hot(difficulties, k)
    embeddings = []
    for i in range(len(difficulties)):
        sem_block = np.asarray(sem_vecs[i], dtype=np.float64)
        sem_norm = np.linalg.norm(sem_block)
        if sem_norm > 0:
            sem_block = sem_block / sem_norm

        completion_block = np.asarray(completion_vecs[i], dtype=np.float64)
        completion_norm = np.linalg.norm(completion_block)
        if completion_norm > 0:
            completion_block = completion_block / completion_norm

        combined = np.concatenate([sem_block, completion_block, diff_blocks[i]])
        embeddings.append(combined.astype(np.float32))
    return embeddings
