import unittest

from core_llm.tokenizer import XNLPTokenizer


class TokenizerRegressionTests(unittest.TestCase):
    def make_tokenizer(self, texts):
        tokenizer = XNLPTokenizer(vocab_size=512, min_frequency=1)
        tokenizer.train(texts, verbose=False)
        return tokenizer

    def assert_round_trip(self, tokenizer, text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_isi_xhosa_sentence_boundaries(self):
        texts = [
            "Umntu ngumntu ngabantu.",
            "Ubuntu buhle.",
            "Ndiyafunda isiXhosa.",
            "Ityala lamawele.",
            "AmaXhosa anembali ende.",
        ]
        tokenizer = self.make_tokenizer(texts)
        for text in texts:
            with self.subTest(text=text):
                self.assert_round_trip(tokenizer, text)

    def test_supported_whitespace_and_punctuation(self):
        texts = [
            "Igama  lam nguMzi.",
            "Ndingubani? Ndingumfundi!",
            "Ndithi 'molo' kuye.",
            "Umntu\nngumntu ngabantu.",
            "ukuguquguquka-nokwakha",
            "singleword",
            "",
        ]
        tokenizer = self.make_tokenizer(texts)
        for text in texts:
            with self.subTest(text=repr(text)):
                self.assert_round_trip(tokenizer, text)

    def test_special_tokens_are_skipped_without_dropping_text(self):
        text = "Ubuntu buhle."
        tokenizer = self.make_tokenizer([text])
        ids = tokenizer.encode(text, add_special_tokens=True)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertNotIn(tokenizer.bos_token_id, tokenizer.encode(text, add_special_tokens=False))

    def test_merge_indexing_is_deterministic_and_order_preserving(self):
        texts = ["ab ab ab", "aba bab", "ababa"]
        first = self.make_tokenizer(texts)
        second = self.make_tokenizer(texts)
        for text in texts:
            self.assertEqual(first.encode(text, add_special_tokens=False), second.encode(text, add_special_tokens=False))
            self.assertEqual(first.decode(first.encode(text, add_special_tokens=False)), text)
        self.assertEqual(first.merges, second.merges)

    def test_tokenizer_state_round_trips_through_serialization(self):
        """Tokenizer state must survive a checkpoint save/load cycle intact.

        The single-file checkpoint design embeds the tokenizer state inside the
        ``.pt`` blob via ``tokenizer_to_state_dict`` / ``tokenizer_from_state_dict``.
        This is currently untested; it must preserve the vocabulary, merges and
        encode/decode behaviour exactly.
        """
        from xnlp_trainer.data import tokenizer_to_state_dict, tokenizer_from_state_dict
        texts = ["Umntu ngumntu ngabantu.", "Ubuntu buhle.", "Ndiyafunda isiXhosa."]
        tokenizer = self.make_tokenizer(texts)
        state = tokenizer_to_state_dict(tokenizer)
        restored = tokenizer_from_state_dict(state)

        self.assertEqual(restored.token2id, tokenizer.token2id)
        self.assertEqual(restored.merges, tokenizer.merges)
        self.assertEqual(restored.merge_ranks, tokenizer.merge_ranks)
        self.assertEqual(restored.num_merges, tokenizer.num_merges)

        for text in texts:
            self.assertEqual(
                tokenizer.encode(text, add_special_tokens=False),
                restored.encode(text, add_special_tokens=False),
            )
            encoded = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(restored.decode(encoded), text)
            self.assertEqual(tokenizer.decode(encoded), restored.decode(encoded))


class WhitespaceHandlingTests(unittest.TestCase):
    """Root-cause regression tests for whitespace preservation (fast, no training).

    These pin the two defects that previously broke round-tripping on small
    toy corpora:

      1. ``_pre_tokenize`` discarded inter-word whitespace via ``text.split()``,
         so whitespace never entered the vocabulary and encode/decode could not
         reconstruct it.
      2. ``decode`` collapsed whitespace runs by default, destroying multiple
         spaces and newlines.
    """

    def make_minimal_tokenizer(self):
        tok = XNLPTokenizer(vocab_size=64, min_frequency=1)
        # Seed a space character and a normal letter into the vocabulary so
        # encode/decode can exercise whitespace handling without running full
        # BPE training.
        if " " not in tok.token2id:
            idx = len(tok.token2id)
            tok.token2id[" "] = idx
            tok.id2token[idx] = " "
        if "x" not in tok.token2id:
            idx = len(tok.token2id)
            tok.token2id["x"] = idx
            tok.id2token[idx] = "x"
        return tok

    def test_pre_tokenize_preserves_whitespace_runs(self):
        tok = self.make_minimal_tokenizer()
        # Whitespace runs (single, double, newline) must be kept as chunks
        # instead of being discarded by a naive ``text.split()``.
        self.assertEqual(
            tok._pre_tokenize("a  b\n c"),
            ["a", "  ", "b", "\n ", "c"],
        )
        self.assertEqual(tok._pre_tokenize("  lead"), ["  ", "lead"])
        self.assertEqual(tok._pre_tokenize("word"), ["word"])
        self.assertEqual(tok._pre_tokenize(""), [])

    def test_decode_does_not_collapse_whitespace_by_default(self):
        tok = self.make_minimal_tokenizer()
        x, s = tok.token2id["x"], tok.token2id[" "]
        ids = [x, s, s, x]
        # Default must preserve multiple spaces (exact round-trip contract).
        self.assertEqual(tok.decode(ids), "x  x")
        # Opt-in cleanup collapses the run to a single space.
        self.assertEqual(tok.decode(ids, clean_up_tokenization_spaces=True), "x x")


if __name__ == "__main__":
    unittest.main()
