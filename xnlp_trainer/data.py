"""
XNLP Trainer – Data Pipeline
=============================
Corpus loading, BPE tokenizer training, dataset construction and
train/val/test splitting.

Everything needed to turn raw ``.txt`` corpus files into PyTorch
``DataLoader`` objects is here.  Tokenizer state is serialisable so it
can be embedded inside the single self-contained checkpoint file.
"""

from __future__ import annotations

import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from core_llm.tokenizer import XNLPTokenizer, SpecialTokens

# Re-export for convenience
__all__ = [
    "load_corpus",
    "corpus_fingerprint",
    "train_tokenizer",
    "tokenize_texts",
    "build_split_indices",
    "XhosaTextDataset",
    "make_collate_fn",
    "prepare_data",
    "tokenizer_to_state_dict",
    "tokenizer_from_state_dict",
]


# ─── Corpus loading ─────────────────────────────────────────────────────────


def load_corpus(corpus_dir: str) -> List[str]:
    """Load every ``.txt`` file in *corpus_dir* and return non-empty lines."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_sentences: List[str] = []
    for txt_file in sorted(corpus_path.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        for line in text.splitlines():
            s = line.strip()
            if s:
                all_sentences.append(s)

    # De-duplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def corpus_fingerprint(corpus_dir: str) -> str:
    """Return a stable SHA-256 fingerprint of the corpus inputs.

    File names and raw UTF-8 bytes are hashed in sorted path order. This makes
    the fingerprint independent of filesystem enumeration order and lets
    checkpoints detect accidental resume against a changed corpus.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    digest = hashlib.sha256()
    files = sorted(corpus_path.glob("*.txt"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# ─── Tokenizer training ────────────────────────────────────────────────────


def train_tokenizer(
    sentences: List[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
    verbose: bool = True,
) -> XNLPTokenizer:
    """Train a fresh BPE tokenizer on the given sentences."""
    if verbose:
        print(f"  Training BPE tokenizer on {len(sentences)} sentences "
              f"(target vocab {vocab_size})...")
    tokenizer = XNLPTokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.train(sentences, verbose=verbose)
    tokenizer.training_data_size = len(sentences)
    return tokenizer


# ─── Tokenizer serialisation (for single-file checkpoints) ──────────────────


def tokenizer_to_state_dict(tokenizer: XNLPTokenizer) -> Dict[str, Any]:
    """Serialise the full tokenizer state into a plain dict."""
    st = tokenizer.special_tokens
    return {
        "tokenizer_type": "xnlp_bpe",
        "tokenizer_version": "2.0.0",
        "vocab_size": tokenizer.vocab_size,
        "min_frequency": tokenizer.min_frequency,
        "num_merges": tokenizer.num_merges,
        "training_data_size": tokenizer.training_data_size,
        "special_tokens": {
            "pad_token": st.pad_token,
            "bos_token": st.bos_token,
            "eos_token": st.eos_token,
            "unk_token": st.unk_token,
            "mask_token": st.mask_token,
            "sep_token": st.sep_token,
            "cls_token": st.cls_token,
        },
        "token2id": tokenizer.token2id,
        "id2token": tokenizer.id2token,
        "merges": [list(m) for m in tokenizer.merges],
        "merge_ranks": {f"\0{pair[0]}\0{pair[1]}": rank
                        for pair, rank in tokenizer.merge_ranks.items()},
    }


def tokenizer_from_state_dict(state: Dict[str, Any]) -> XNLPTokenizer:
    """Reconstruct an ``XNLPTokenizer`` from a serialised state dict."""
    st_cfg = state["special_tokens"]
    special_tokens = SpecialTokens(
        pad_token=st_cfg["pad_token"],
        bos_token=st_cfg["bos_token"],
        eos_token=st_cfg["eos_token"],
        unk_token=st_cfg["unk_token"],
        mask_token=st_cfg.get("mask_token", ""),
        sep_token=st_cfg.get("sep_token", "[SEP]"),
        cls_token=st_cfg.get("cls_token", "[CLS]"),
    )
    tokenizer = XNLPTokenizer(
        vocab_size=state["vocab_size"],
        special_tokens=special_tokens,
        min_frequency=state["min_frequency"],
    )
    tokenizer.token2id = state["token2id"]
    tokenizer.id2token = state["id2token"]
    tokenizer.merges = [tuple(m) for m in state["merges"]]
    tokenizer.merge_ranks = {
        (k.split("\0")[1], k.split("\0")[2]): v
        for k, v in state["merge_ranks"].items()
    }
    tokenizer.num_merges = state["num_merges"]
    tokenizer.training_data_size = state.get("training_data_size", 0)
    return tokenizer


# ─── Tokenisation helpers ──────────────────────────────────────────────────


def tokenize_texts(
    sentences: List[str],
    tokenizer: XNLPTokenizer,
    max_len: int,
    add_special_tokens: bool = True,
) -> List[List[int]]:
    """Tokenise a list of raw sentences into ID sequences."""
    return [
        tokenizer.encode(
            s, add_special_tokens=add_special_tokens,
            max_length=max_len, truncation=True,
        )
        for s in sentences
    ]


def build_split_indices(
    n: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return shuffled train/val/test index lists for *n* items."""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ─── Dataset & collation ────────────────────────────────────────────────────


class XhosaTextDataset(Dataset):
    """Tokenised text dataset for causal LM training."""

    def __init__(self, tokenised_ids: List[List[int]], max_len: int):
        self.max_len = max_len
        # keep only sequences that are at least 4 tokens
        self.examples = [ids[:max_len] for ids in tokenised_ids if len(ids) >= 4]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


def make_collate_fn(pad_token_id: int, max_len: int):
    """Return a collate function using **dynamic padding**.

    Each batch is padded to the length of the longest sequence in that
    batch (capped at *max_len*), not to *max_len* unconditionally.  This
    avoids wasting compute on padding tokens — critical on a CPU-only
    machine where the average V2 sentence is ~18 tokens but the cap is
    64.  Returns an ``attention_mask`` so the model can skip padding
    positions during attention.
    """

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Dynamic padding: pad to the longest sequence in the batch
        actual_max = min(max(len(b["input_ids"]) for b in batch), max_len)
        n = len(batch)
        input_ids = torch.full((n, actual_max), pad_token_id, dtype=torch.long)
        labels = torch.full((n, actual_max), -100, dtype=torch.long)
        attention_mask = torch.zeros((n, actual_max), dtype=torch.long)
        for i, b in enumerate(batch):
            length = min(len(b["input_ids"]), actual_max)
            input_ids[i, :length] = b["input_ids"][:length]
            labels[i, :length] = b["labels"][:length]
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attention_mask}

    return collate


# ─── High-level data preparation ────────────────────────────────────────────


def prepare_data(
    config,
    tokenizer: Optional[XNLPTokenizer] = None,
    verbose: bool = True,
    include_test: bool = False,
) -> Tuple:
    """
    Load corpus → train/load tokenizer → build train & val loaders.

    Parameters
    ----------
    config : TrainingConfig or dict
        Must have attributes ``corpus_dir``, ``vocab_size``,
        ``tokenizer_min_freq``, ``max_seq_len``, ``train_split``,
        ``val_split``, ``test_split``, ``seed``, ``batch_size``,
        ``num_workers``.
    tokenizer : XNLPTokenizer, optional
        If provided, skip tokenizer training and use this one.
    """
    # Allow config to be a dict for flexibility
    if isinstance(config, dict):
        cfg = type("Cfg", (), config)()
    else:
        cfg = config

    if verbose:
        print(f"[data] Loading corpus from '{cfg.corpus_dir}' ...")
    sentences = load_corpus(cfg.corpus_dir)
    if not sentences:
        raise RuntimeError(f"No sentences found in {cfg.corpus_dir}")
    if verbose:
        print(f"[data] {len(sentences)} unique sentences loaded")

    # ── Tokeniser ──────────────────────────────────────────────────────
    if tokenizer is None:
        train_idx, _, _ = build_split_indices(
            len(sentences), cfg.train_split, cfg.val_split, cfg.seed,
        )
        tokenizer = train_tokenizer(
            [sentences[i] for i in train_idx],
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.tokenizer_min_freq,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"[data] Using pre-trained tokenizer "
                  f"(vocab={tokenizer.vocab_size_actual})")

    pad_id = tokenizer.pad_token_id

    # ── Split sentences ────────────────────────────────────────────────
    train_idx, val_idx, test_idx = build_split_indices(
        len(sentences), cfg.train_split, cfg.val_split, cfg.seed,
    )
    train_sents = [sentences[i] for i in train_idx]
    val_sents = [sentences[i] for i in val_idx]
    test_sents = [sentences[i] for i in test_idx]

    # ── Tokenise ───────────────────────────────────────────────────────
    if verbose:
        print("[data] Tokenising splits ...")
    train_ids = tokenize_texts(train_sents, tokenizer, cfg.max_seq_len)
    val_ids = tokenize_texts(val_sents, tokenizer, cfg.max_seq_len)
    test_ids = tokenize_texts(test_sents, tokenizer, cfg.max_seq_len)

    # ── Datasets ───────────────────────────────────────────────────────
    train_ds = XhosaTextDataset(train_ids, cfg.max_seq_len)
    val_ds = XhosaTextDataset(val_ids, cfg.max_seq_len)
    test_ds = XhosaTextDataset(test_ids, cfg.max_seq_len)
    if verbose:
        print(f"[data] Train examples: {len(train_ds)} | "
              f"Val examples: {len(val_ds)} | Test examples: {len(test_ds)}")

    # ── DataLoader ─────────────────────────────────────────────────────
    collate = make_collate_fn(pad_id, cfg.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    if include_test:
        return train_loader, val_loader, test_loader, tokenizer
    return train_loader, val_loader, tokenizer
