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


if __name__ == "__main__":
    unittest.main()
