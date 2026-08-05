"""Teacher client: Qwen3-32B on AWS Bedrock via LangChain."""

from __future__ import annotations

from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse

DEFAULT_MODEL_ID = "qwen.qwen3-32b-v1:0"


class TeacherClient:
    """Batched text-completion wrapper around a Bedrock-hosted chat model."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id

    def generate(self, prompts: list[str], *, max_tokens: int, temperature: float) -> list[str]:
        """Generate one completion per prompt, in the same order as the input."""
        llm = ChatBedrockConverse(
            model=self._model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            # llm.batch() dispatches prompts concurrently with no cap tied to botocore's default
            # pool size (10), so large batches otherwise churn through discarded/reopened
            # connections. Size the pool to the batch so it's never exhausted.
            config=BotocoreConfig(max_pool_connections=max(10, len(prompts))),
        )
        responses = llm.batch(prompts)
        return [r.content for r in responses]
