import pytest

from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.data import build_split_indices


def test_bpe_merges_the_selected_occurrence_not_first_symbol():
    tokenizer = XNLPTokenizer(vocab_size=32, min_frequency=1)
    tokenizer.token2id["a"] = 5
    tokenizer.id2token[5] = "a"
    tokenizer.token2id["b"] = 6
    tokenizer.id2token[6] = "b"
    tokenizer.token2id["ab"] = 7
    tokenizer.id2token[7] = "ab"
    tokenizer.merges = [("a", "b")]
    tokenizer.merge_ranks = {("a", "b"): 0}
    tokenizer.num_merges = 1

    # The selected pair is the second "a" + "b" in "aab".
    ids = tokenizer._encode_word("aab")

    assert ids == [
        tokenizer.token2id["a"],
        tokenizer.token2id["ab"],
    ]


def test_split_indices_are_disjoint_and_cover_all_examples():
    train, val, test = build_split_indices(101, 0.8, 0.15, seed=42)

    assert len(train) == 80
    assert len(val) == 15
    assert len(test) == 6

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(range(101))


def test_split_indices_are_reproducible():
    first = build_split_indices(101, 0.8, 0.15, seed=42)
    second = build_split_indices(101, 0.8, 0.15, seed=42)
    assert first == second
