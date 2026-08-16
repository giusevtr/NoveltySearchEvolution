"""Student client: colocated vLLM inference for Qwen3-0.6B.

Adapted from temp/utils/vllm_inference_server/model_manager.py::ModelManager, trimmed to
just load()/generate() — no LoRA, no teacher-forced perplexity scoring (not needed here;
difficulty is measured by sampling, see novelty_search/embedding.py and run.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm import LLM, SamplingParams  # type: ignore
from vllm.lora.request import LoRARequest  # type: ignore

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class ModelLoadError(RuntimeError):
    """Raised when the student model fails to load."""


@dataclass
class StudentConfig:
    name: str = DEFAULT_MODEL_NAME
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 2048
    lora_path: str | None = None
    max_lora_rank: int = 16


class StudentClient:
    """Wraps vllm.LLM for colocated student sampling."""

    def __init__(self, config: StudentConfig | None = None) -> None:
        self._config = config or StudentConfig()
        self._llm: LLM | None = None
        self._lora_request: LoRARequest | None = None

    def load(self) -> None:
        if self._llm is not None:
            return
        try:
            lora_enabled = self._config.lora_path is not None
            llm_kwargs = {}
            if lora_enabled:
                llm_kwargs["enable_lora"] = True
                llm_kwargs["max_lora_rank"] = self._config.max_lora_rank
            self._llm = LLM(
                model=self._config.name,
                dtype=self._config.dtype,
                gpu_memory_utilization=self._config.gpu_memory_utilization,
                max_model_len=self._config.max_model_len,
                **llm_kwargs,
            )
            self._lora_request = (
                LoRARequest("adapter", 1, self._config.lora_path) if lora_enabled else None
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load student model {self._config.name}: {e}") from e

    def format_chat_prompt(self, user_content: str) -> str:
        """Wrap a user message in the model's chat template (assistant-turn-ready).

        enable_thinking=False skips Qwen3's <think>...</think> preamble so it answers directly
        instead of rambling/self-correcting in an open-ended "thinking" continuation.
        """
        if self._llm is None:
            raise RuntimeError("Student model not loaded; call load() first")
        tokenizer = self._llm.get_tokenizer()
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(
        self, prompts: list[str], *, n: int, temperature: float, max_tokens: int
    ) -> list[list[str]]:
        """Generate `n` sampled completions per prompt.

        Returns:
            List (len == len(prompts)) of lists (len == n) of completion strings.
        """
        if self._llm is None:
            raise RuntimeError("Student model not loaded; call load() first")

        chat_prompts = [self.format_chat_prompt(p) for p in prompts]
        params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
        outputs = self._llm.generate(chat_prompts, params, lora_request=self._lora_request)
        if len(outputs) != len(chat_prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} results for {len(chat_prompts)} prompts; "
                "results can no longer be aligned with their questions"
            )
        return [[completion.text for completion in out.outputs] for out in outputs]
