"""Fine-tune an existing XNLP checkpoint on structure, grammar, QA and continuation."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from xnlp_trainer.config import validate_checkpoint
from xnlp_trainer.data import tokenizer_from_state_dict
from xnlp_trainer.structured_curriculum import build_curriculum


class StructuredDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int = 256):
        self.items = []
        self.item_tasks = []
        self.dropped_long_targets = 0
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
            full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            if len(full_ids) > max_length:
                self.dropped_long_targets += 1
                continue
            ids = full_ids
            target_start = len(prompt_ids)
            labels = [-100] * target_start + ids[target_start:]
            labels = labels[:max_length]
            if len(ids) >= 3 and any(label != -100 for label in labels):
                self.items.append((ids, labels))
                self.item_tasks.append(row.get("task", "UNKNOWN"))
        self.task_counts = {}
        for task in self.item_tasks:
            self.task_counts[task] = self.task_counts.get(task, 0) + 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ids, labels = self.items[index]
        return torch.tensor(ids), torch.tensor(labels)


def _collate(batch, pad_id: int):
    length = max(len(ids) for ids, _ in batch)
    inputs = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), length), dtype=torch.long)
    for i, (ids, target) in enumerate(batch):
        inputs[i, :len(ids)] = ids
        labels[i, :len(target)] = target
        mask[i, :len(ids)] = 1
    return inputs, labels, mask


def _model_from_checkpoint(ckpt: dict[str, Any], device: str, max_position_embeddings: int | None = None):
    cfg = ckpt["config"]
    config = XNLPConfig(
        vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=max_position_embeddings or cfg["max_position_embeddings"], device=device,
    )
    model = XNLPCoreLLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def train_structured(checkpoint: str, corpus: str, output: str, *, max_steps: int = 200,
                     batch_size: int = 4, learning_rate: float = 5e-5, seed: int = 42,
                     max_length: int = 512, curriculum_path: str | None = None,
                     checkpoint_interval: int = 0) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(ckpt)
    tokenizer = tokenizer_from_state_dict(ckpt["tokenizer_state"])
    curriculum_file = Path(curriculum_path) if curriculum_path else Path(output).with_suffix(".jsonl")
    if curriculum_path:
        if not curriculum_file.exists():
            raise FileNotFoundError(f"Curriculum file does not exist: {curriculum_file}")
        curriculum = {"source": str(curriculum_file), "examples": sum(1 for _ in curriculum_file.open(encoding="utf-8")),
                      "counts": {}, "output": str(curriculum_file), "reused": True}
    else:
        curriculum = build_curriculum(corpus, curriculum_file)
    rows = [json.loads(line) for line in curriculum_file.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        curriculum["counts"][row.get("task", "UNKNOWN")] = curriculum["counts"].get(row.get("task", "UNKNOWN"), 0) + 1
    dataset = StructuredDataset(rows, tokenizer, max_length)
    if not dataset:
        raise ValueError(
            "Structured curriculum has no examples within max_length; "
            f"dropped_long_targets={dataset.dropped_long_targets}, max_length={max_length}"
        )
    # The full canonical structure target is rarer after context filtering.
    # Sample tasks uniformly so the model cannot spend nearly every update on
    # the shorter grammar/QA/continuation targets.
    task_max = max(dataset.task_counts.values())
    sample_weights = [1.0 / dataset.task_counts[task] for task in dataset.item_tasks]
    sampler = WeightedRandomSampler(sample_weights, num_samples=task_max * len(dataset.task_counts), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id))
    context_length = max(max_length, int(ckpt["config"]["max_position_embeddings"]))
    model = _model_from_checkpoint(ckpt, device, context_length)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    losses = []
    step = 0
    while step < max_steps:
        for inputs, labels, mask in loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            loss = model(input_ids=inputs, attention_mask=mask, labels=labels)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if checkpoint_interval and step % checkpoint_interval == 0:
                partial = dict(ckpt)
                partial["model_state_dict"] = model.state_dict()
                partial["config"]["max_position_embeddings"] = context_length
                partial["training_metadata"] = dict(
                    ckpt["training_metadata"], structured_finetune_steps=step,
                    structured_curriculum=curriculum, structured_learning_rate=learning_rate,
                    partial_checkpoint=True,
                )
                partial_path = Path(output).with_suffix(f".step{step}.pt")
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(partial, partial_path)
            if step >= max_steps:
                break
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["config"]["max_position_embeddings"] = context_length
    ckpt["training_metadata"] = dict(ckpt["training_metadata"], structured_finetune_steps=step,
                                      structured_curriculum=curriculum, structured_learning_rate=learning_rate)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    report = {"checkpoint": str(output_path), "device": device, "steps": step,
              "examples": len(dataset), "dropped_long_targets": dataset.dropped_long_targets,
              "initial_loss": losses[0], "final_loss": losses[-1],
              "mean_loss": sum(losses) / max(len(losses), 1), "task_counts": dataset.task_counts,
              "sampling": "uniform_over_retained_tasks", "curriculum": curriculum}
    output_path.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/xnlp_prose_pilot/model/best_model.pt")
    parser.add_argument("--corpus", default="data/processed/v2/splits/train.txt")
    parser.add_argument("--output", default="artifacts/xnlp_structured/model/structured_model.pt")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--curriculum", default=None)
    args = parser.parse_args()
    print(json.dumps(train_structured(args.checkpoint, args.corpus, args.output, max_steps=args.max_steps,
                                      max_length=args.max_length, batch_size=args.batch_size,
                                      curriculum_path=args.curriculum,
                                      checkpoint_interval=args.checkpoint_interval),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
