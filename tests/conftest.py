"""Shared pytest fixtures and mocks for test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_vllm_output():
    """Mock object mimicking vLLM RequestOutput."""
    output = MagicMock()
    output.prompt = "Hello, how are you?"
    output.prompt_token_ids = [1, 2, 3, 4, 5]

    completion = MagicMock()
    completion.text = "I'm doing well, thank you!"
    completion.token_ids = [6, 7, 8, 9, 10, 11, 12]
    output.outputs = [completion]

    return output


@pytest.fixture
def mock_vllm_llm(monkeypatch, mock_vllm_output):
    """Mock vllm.LLM class."""
    mock_llm_class = MagicMock()
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = [mock_vllm_output]
    mock_llm_class.return_value = mock_llm_instance

    # Patch at the usage site (model_manager module)
    monkeypatch.setattr(
        "src.utils.vllm_inference_server.model_manager.LLM", mock_llm_class
    )

    return mock_llm_class, mock_llm_instance


@pytest.fixture
def mock_torch_cuda(monkeypatch):
    """Mock torch.cuda methods for VRAM tracking."""

    def mock_memory_allocated():
        return 5_000_000_000  # 5GB

    def mock_memory_reserved():
        return 6_000_000_000  # 6GB

    def mock_get_device_properties(device):
        props = MagicMock()
        props.total_memory = 12_000_000_000  # 12GB
        return props

    monkeypatch.setattr("torch.cuda.memory_allocated", mock_memory_allocated)
    monkeypatch.setattr("torch.cuda.memory_reserved", mock_memory_reserved)
    monkeypatch.setattr("torch.cuda.get_device_properties", mock_get_device_properties)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
