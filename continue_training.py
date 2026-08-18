"""Continue training from checkpoint - faster settings for more epochs."""
import sys, os, json, math, time, random
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer


def load_corpus(corpus_dir):
    all_sentences = []
    for txt_file in sorted(Path(corpus_dir).glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        all_sentences.extend(sentences)
    return all_sentences


class XhosaTextDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for sent in sentences:
            ids = tokenizer.encode(sent, add_special_tokens=True)
            if len(ids) < 4:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx][:self.max_len]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long)}


def collate_fn(batch, pad_id=0, max_len=256):
    padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        l = min(len(b["input_ids"]), max_len)
        padded_inputs[i, :l] = b["input_ids"][:l]
        padded_labels[i, :l] = b["labels"][:l]
    return {"input_ids": padded_inputs, "labels": padded_labels}


def get_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = XNLPTokenizer.load_pretrained("trained_model/tokenizer")
    vocab_actual = tokenizer.vocab_size_actual
    print(f"Vocab: {vocab_actual}")

    # Load corpus
    sentences = load_corpus("corpus")
    print(f"Corpus: {len(sentences)} sentences")

    # Dataset
    dataset = XhosaTextDataset(sentences, tokenizer, max_len=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                       collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, 256),
                       num_workers=0)
    print(f"Dataset: {len(dataset)} examples, {len(loader)} batches/epoch")

    # Load model from checkpoint
    ckpt = torch.load("trained_model/model_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    config = XNLPConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        max_position_embeddings=cfg["max_seq_len"],
        dropout_prob=0.1,
        device=device,
    )
    model = XNLPCoreLLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params | Epoch {ckpt['epoch']} | Loss {ckpt['loss']:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    max_epochs = 30
    max_steps = max_epochs * len(loader)
    warmup = min(200, len(loader) * 2)

    print(f"\nTraining {max_epochs} more epochs ({max_steps} steps)...")
    print("=" * 60)

    global_step = 0
    start_epoch = ckpt["epoch"]

    for epoch in range(start_epoch, start_epoch + max_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        t0 = time.time()

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(global_step, warmup, max_steps, 3e-4)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1
            global_step += 1

            if global_step % 20 == 0:
                print(f"  Ep {epoch+1} Step {global_step} Loss {loss.item():.4f} Avg {epoch_loss/n:.4f} LR {lr:.2e}")

        avg = epoch_loss / max(n, 1)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1} complete | Loss: {avg:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        save_path = f"trained_model/model_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss": avg,
            "config": cfg,
        }, save_path)
        print(f"  Saved: {save_path}")

    # Final save
    torch.save({
        "epoch": start_epoch + max_epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg,
        "config": cfg,
    }, "trained_model/best_model.pt")
    print(f"\nFinal model saved: trained_model/best_model.pt")

    # Quick test
    model.eval()
    prompts = ["Umntu ngumntu", "Imbongi yethu", "Ityala lamawele"]
    print("\n--- Generation Test ---")
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=20, temperature=0.8, top_k=40)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"  {p} -> {text}")

    print("=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
