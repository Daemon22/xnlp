import pytest
import torch

from core_llm.architecture import XNLPConfig, XNLPCoreLLM

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


def test_kv_cache_matches_full_context_logits():
    torch.manual_seed(7)
    config = XNLPConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        dropout_prob=0.0,
    )
    model = XNLPCoreLLM(config).eval()
    prompt = torch.tensor([[1, 5, 8, 13]], dtype=torch.long)
    next_token = torch.tensor([[21]], dtype=torch.long)

    full = model(
        input_ids=torch.cat([prompt, next_token], dim=1),
        use_cache=False,
    )
    cached_prompt = model(
        input_ids=prompt,
        use_cache=True,
    )
    cached_next = model(
        input_ids=next_token,
        past_key_values=cached_prompt["past_key_values"],
        use_cache=True,
    )

    assert torch.allclose(
        full["logits"][:, -1, :],
        cached_next["logits"][:, -1, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_generation_supports_batched_greedy_decoding():
    torch.manual_seed(11)
    config = XNLPConfig(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        dropout_prob=0.0,
    )
    model = XNLPCoreLLM(config).eval()
    prompts = torch.tensor([[1, 4, 7], [1, 9, 10]], dtype=torch.long)

    generated = model.generate(
        prompts,
        max_new_tokens=3,
        do_sample=False,
        repetition_penalty=1.1,
    )

    assert generated.shape == (2, 6)
    assert torch.equal(generated[:, :3], prompts)


def test_generation_rejects_context_overflow():
    config = XNLPConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4,
        dropout_prob=0.0,
    )
    model = XNLPCoreLLM(config).eval()

    with pytest.raises(ValueError, match="context window"):
        model.generate(torch.tensor([[1, 2, 3]], dtype=torch.long), max_new_tokens=2)
