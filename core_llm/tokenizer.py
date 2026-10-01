"""
XNLP Core LLM Tokenizer
========================
Enhanced BPE tokenizer with special tokens for LLM functionality.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class SpecialTokens:
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    unk_token: str = "[UNK]"
    mask_token: str = ""
    sep_token: str = "[SEP]"
    cls_token: str = "[CLS]"
    
    @property
    def all_tokens(self) -> List[str]:
        return [self.pad_token, self.bos_token, self.eos_token, self.unk_token,
                self.mask_token, self.sep_token, self.cls_token]
    
    @property
    def token_to_id(self) -> Dict[str, int]:
        return {token: idx for idx, token in enumerate(self.all_tokens)}
    
    @property
    def id_to_token(self) -> Dict[int, str]:
        return {idx: token for idx, token in enumerate(self.all_tokens)}
    
    @property
    def num_special_tokens(self) -> int:
        return len(self.all_tokens)


class XNLPTokenizer:
    def __init__(self, vocab_size=8000, special_tokens=None, min_frequency=2):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SpecialTokens()
        self.min_frequency = min_frequency
        
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        
        self._init_special_tokens()
        self.num_merges = 0
        self.training_data_size = 0
    
    def _init_special_tokens(self):
        for token in self.special_tokens.all_tokens:
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens.token_to_id[self.special_tokens.pad_token]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens.token_to_id[self.special_tokens.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens.token_to_id[self.special_tokens.eos_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens.token_to_id[self.special_tokens.unk_token]
    
    @property
    def vocab_size_actual(self) -> int:
        return len(self.token2id)
    
    def train(self, texts, verbose=True):
        if verbose:
            print(f"Training XNLP Tokenizer on {len(texts)} texts...")
        
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                symbols = tuple(word)
                word_freqs[symbols] = word_freqs.get(symbols, 0) + 1
        
        if verbose:
            print(f"Found {len(word_freqs)} unique words")
        
        self._init_base_vocab(word_freqs)
        
        if verbose:
            print(f"Base vocabulary size: {len(self.token2id)}")
        
        max_merges = self.vocab_size - len(self.token2id)
        
        for merge_idx in range(max_merges):
            best_pair = None
            best_count = 0
            
            for pair, count in self._count_pairs(word_freqs).items():
                if count >= self.min_frequency and count > best_count:
                    best_pair = pair
                    best_count = count
            
            if best_pair is None:
                if verbose:
                    print(f"No more valid merges at iteration {merge_idx}")
                break
            
            self._merge_pair(best_pair, word_freqs)
            
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merge_ranks)
            self.num_merges += 1
            
            if new_token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[new_token] = idx
                self.id2token[idx] = new_token
            
            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"  Merge {merge_idx + 1}: {best_pair} -> {new_token} (vocab: {len(self.token2id)})")
        
        if verbose:
            print(f"Training complete! Final vocab size: {len(self.token2id)}, Merges: {self.num_merges}")
    
    def _pre_tokenize(self, text):
        # Split into alternating non-whitespace "words" and whitespace runs.
        # Whitespace is preserved as its own characters (rather than discarded
        # via ``text.split()``) so that:
        #   * the space/newline/tab characters enter the vocabulary, and
        #   * encode / decode round-trip the original surface text exactly,
        #     including multiple consecutive spaces and newlines.
        # Discarding inter-word whitespace makes round-tripping impossible
        # and causes generated text to run words together ("umntungumntu").
        result = []
        for chunk in re.findall(r"\S+|\s+", text):
            parts = re.findall(r"([^\w]*)(\w+)([^\w]*)", chunk)
            if parts:
                for prefix, core, suffix in parts:
                    if prefix:
                        result.append(prefix)
                    result.append(core)
                    if suffix:
                        result.append(suffix)
            else:
                # Whitespace (or punctuation-only) chunk: keep verbatim so its
                # characters survive into the vocabulary and round-trip.
                result.append(chunk)
        return result
    
    def _init_base_vocab(self, word_freqs):
        for word_tuple in word_freqs.keys():
            for char in word_tuple:
                if char not in self.token2id:
                    idx = len(self.token2id)
                    self.token2id[char] = idx
                    self.id2token[idx] = char
    
    def _count_pairs(self, word_freqs):
        pairs: Dict[Tuple[str, str], int] = {}
        for word_tuple, freq in word_freqs.items():
            if len(word_tuple) < 2:
                continue
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def _merge_pair(self, pair, word_freqs):
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        merged = pair[0] + pair[1]
        
        for word_tuple, freq in word_freqs.items():
            new_word: List[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def encode(self, text, add_special_tokens=True, max_length=None,
               truncation=True, padding=False, return_tensors=False):
        tokens: List[int] = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        words = self._pre_tokenize(text)
        for word in words:
            encoded_word = self._encode_word(word)
            tokens.extend(encoded_word)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        if max_length and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.eos_token_id]
        
        if padding and max_length:
            while len(tokens) < max_length:
                tokens.append(self.pad_token_id)
        
        if return_tensors:
            import torch
            return torch.tensor(tokens, dtype=torch.long)
        
        return tokens
    
    def _encode_word(self, word):
        """Encode one word using deterministic, occurrence-safe BPE merges."""
        if not word:
            return []

        tokens = list(word)
        while len(tokens) > 1:
            best_pair = None
            best_index = -1
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_index = i

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            tokens = tokens[:best_index] + [merged] + tokens[best_index + 2:]

        ids = []
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.extend(self._fallback_encode(token))
        return ids

    def _fallback_encode(self, token):
        ids = []
        for char in token:
            if char in self.token2id:
                ids.append(self.token2id[char])
            else:
                ids.append(self.unk_token_id)
        return ids
    
    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        tokens = []
        for tid in token_ids:
            if tid in self.id2token:
                token = self.id2token[tid]
                if skip_special_tokens and token in self.special_tokens.all_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.special_tokens.unk_token)

        text = "".join(tokens)

        # ``clean_up_tokenization_spaces`` collapses runs of whitespace into a
        # single space. Because this tokenizer now preserves whitespace as real
        # tokens, such collapsing would destroy legitimate spacing (e.g. double
        # spaces, newlines) and break exact round-tripping. It is therefore off
        # by default and only applied when a caller explicitly opts in.
        if clean_up_tokenization_spaces:
            text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        with open(os.path.join(save_directory, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_directory, "merges.json"), 'w', encoding='utf-8') as f:
            json.dump([list(m) for m in self.merges], f, ensure_ascii=False, indent=2)
        
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": {
                "pad_token": self.special_tokens.pad_token,
                "bos_token": self.special_tokens.bos_token,
                "eos_token": self.special_tokens.eos_token,
                "unk_token": self.special_tokens.unk_token,
            },
            "num_merges": self.num_merges,
            "model_type": "xnlp_bpe",
            "version": "2.0.0",
        }
        with open(os.path.join(save_directory, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def load_pretrained(cls, load_directory):
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        special_tokens_config = config.get("special_tokens", {})
        special_tokens = SpecialTokens(
            pad_token=special_tokens_config.get("pad_token", "[PAD]"),
            bos_token=special_tokens_config.get("bos_token", "[BOS]"),
            eos_token=special_tokens_config.get("eos_token", "[EOS]"),
            unk_token=special_tokens_config.get("unk_token", "[UNK]"),
        )
        
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 8000),
            special_tokens=special_tokens,
            min_frequency=config.get("min_frequency", 2),
        )
        
        vocab_path = os.path.join(load_directory, "vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.token2id = json.load(f)
        tokenizer.id2token = {v: k for k, v in tokenizer.token2id.items()}
        
        merges_path = os.path.join(load_directory, "merges.json")
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_list = json.load(f)
        tokenizer.merges = [tuple(m) for m in merges_list]
        tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer.merges)}
        tokenizer.num_merges = len(tokenizer.merges)
        
        tokenizer.training_data_size = config.get("training_data_size", 0)
        
        print(f"Tokenizer loaded from {load_directory} (vocab: {tokenizer.vocab_size_actual})")
        return tokenizer
    
    def get_vocab(self):
        return self.token2id.copy()
    
    def __len__(self):
        return len(self.token2id)
    
    def __repr__(self):
        return (f"XNLPTokenizer(vocab_size={self.vocab_size_actual}, "
                f"merges={self.num_merges}, "
                f"special_tokens={self.special_tokens.num_special_tokens})")
