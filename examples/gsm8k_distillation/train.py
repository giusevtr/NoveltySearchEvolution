"""SFT training on a prompt/completion parquet dataset via TRL + LoRA.

Used to train the two distillation arms compared by report.py: the synthetic-data arm (trained
on output/synthetic_distillation.parquet's question/teacher_completion columns) and the
ground-truth baseline arm (trained on a size-matched sample of data/gsm8k_train.parquet's
prompt/completion columns).

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/train.py \
        --data output/synthetic_distillation.parquet \
        --prompt-column question --completion-column teacher_completion \
        --output-dir checkpoints/synthetic

Requires a CUDA GPU. No vLLM engine should be loaded on the same GPU while this runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from examples.gsm8k_distillation.novelty_search.prompts import build_solve_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
DEFAULT_SFT_CONFIG_PATH = THIS_DIR / "configs" / "sft_config.json"
DEFAULT_PEFT_CONFIG_PATH = THIS_DIR / "configs" / "peft_config.json"
DEFAULT_EVAL_DATA_PATH = THIS_DIR / "data" / "gsm8k_val.parquet"

EVALS_PER_EPOCH = 4

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# --- config ---


class SFTHyperparams(BaseModel):
    base_model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_seq_length: int = Field(default=2048, gt=0)
    gradient_checkpointing: bool = True
    micro_batch_size_per_gpu: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    total_epochs: int = Field(default=1, ge=1)


class PeftHyperparams(BaseModel):
    enabled: bool = True
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    task_type: str = "CAUSAL_LM"


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def build_peft_config(cfg: PeftHyperparams) -> LoraConfig | None:
    if not cfg.enabled:
        return None
    return LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=cfg.target_modules,
        task_type=cfg.task_type,
    )


def build_sft_config(
    cfg: SFTHyperparams,
    output_dir: str,
    max_steps: int | None = None,
    report_to: str = "none",
    run_name: str | None = None,
    eval_steps: int | None = None,
) -> SFTConfig:
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.micro_batch_size_per_gpu,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.total_epochs,
        max_length=cfg.max_seq_length,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=(_DTYPE_MAP[cfg.dtype] == torch.bfloat16),
        report_to=report_to,
    )
    if eval_steps is not None:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = eval_steps
        kwargs["per_device_eval_batch_size"] = cfg.micro_batch_size_per_gpu
    else:
        kwargs["eval_strategy"] = "no"
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if run_name is not None:
        kwargs["run_name"] = run_name
    return SFTConfig(**kwargs)


# --- data ---


def load_dataset(
    path: str | Path, prompt_column: str, completion_column: str, max_samples: int | None, seed: int
) -> Dataset:
    frame = pd.read_parquet(path)
    missing = {prompt_column, completion_column} - set(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing column(s) {sorted(missing)}")
    frame = frame[[prompt_column, completion_column]].rename(
        columns={prompt_column: "prompt", completion_column: "completion"}
    )
    if max_samples is not None and max_samples < len(frame):
        frame = frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    # Conversational schema so TRL applies Qwen3's chat template during training, matching
    # StudentClient.format_chat_prompt's enable_thinking=False used at eval time. The user turn
    # is wrapped in build_solve_prompt so it's byte-identical to what eval.py sends at inference
    # time (completions were elicited by this same wrapped prompt, not the bare question).
    frame["prompt"] = frame["prompt"].map(lambda p: [{"role": "user", "content": build_solve_prompt(p)}])
    frame["completion"] = frame["completion"].map(lambda c: [{"role": "assistant", "content": c}])
    frame["chat_template_kwargs"] = [{"enable_thinking": False}] * len(frame)

    return Dataset.from_pandas(frame, preserve_index=False)


# --- train ---


def train(
    data_path: Path,
    prompt_column: str,
    completion_column: str,
    max_samples: int | None,
    output_dir: Path,
    sft_cfg: SFTHyperparams,
    peft_cfg: PeftHyperparams,
    seed: int,
    max_steps: int | None,
    wandb_project: str | None = None,
    run_name: str | None = None,
    eval_data_path: Path | None = None,
    eval_max_samples: int | None = None,
) -> None:
    train_ds = load_dataset(data_path, prompt_column, completion_column, max_samples, seed)
    logger.info("Loaded %d training examples from %s", len(train_ds), data_path)

    eval_ds = None
    eval_steps = None
    if eval_data_path is not None:
        eval_ds = load_dataset(eval_data_path, "prompt", "completion", eval_max_samples, seed)
        logger.info("Loaded %d eval examples from %s", len(eval_ds), eval_data_path)
        steps_per_epoch = math.ceil(
            len(train_ds) / (sft_cfg.micro_batch_size_per_gpu * sft_cfg.gradient_accumulation_steps)
        )
        eval_steps = max(1, steps_per_epoch // EVALS_PER_EPOCH)

    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    dtype = _DTYPE_MAP[sft_cfg.dtype]
    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(sft_cfg.base_model, dtype=dtype)

    peft_config = build_peft_config(peft_cfg)
    training_args = build_sft_config(
        sft_cfg,
        output_dir=str(output_dir),
        max_steps=max_steps,
        report_to="wandb" if wandb_project is not None else "none",
        run_name=run_name,
        eval_steps=eval_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    logger.info("Saved trained model/adapter to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to a prompt/completion parquet file.")
    parser.add_argument("--prompt-column", type=str, default="question")
    parser.add_argument("--completion-column", type=str, default="teacher_completion")
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample the dataset to this many rows (fixed seed).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save the trained LoRA adapter.")
    parser.add_argument("--sft-config", type=Path, default=DEFAULT_SFT_CONFIG_PATH)
    parser.add_argument("--peft-config", type=Path, default=DEFAULT_PEFT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Cap to a handful of samples/steps to verify wiring.")
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="gsm8k-distillation", help="wandb project name (used when --wandb is set).")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for this training arm (used when --wandb is set).")
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA_PATH, help="Path to a prompt/completion parquet file used for periodic eval during training.")
    parser.add_argument("--eval-max-samples", type=int, default=None, help="Subsample the eval dataset to this many rows (fixed seed).")
    parser.add_argument("--no-eval", action="store_true", help="Disable eval-during-training entirely.")
    args = parser.parse_args()

    sft_cfg = SFTHyperparams(**load_json(args.sft_config))
    peft_cfg = PeftHyperparams(**load_json(args.peft_config))

    max_samples = args.max_samples
    max_steps = None
    eval_max_samples = args.eval_max_samples
    if args.smoke_test:
        max_samples = min(8, max_samples) if max_samples is not None else 8
        max_steps = 2
        eval_max_samples = min(8, eval_max_samples) if eval_max_samples is not None else 8

    train(
        data_path=args.data,
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        max_samples=max_samples,
        output_dir=args.output_dir,
        sft_cfg=sft_cfg,
        peft_cfg=peft_cfg,
        seed=args.seed,
        max_steps=max_steps,
        wandb_project=args.wandb_project if args.wandb else None,
        run_name=args.run_name,
        eval_data_path=None if args.no_eval else args.eval_data,
        eval_max_samples=eval_max_samples,
    )


if __name__ == "__main__":
    main()
