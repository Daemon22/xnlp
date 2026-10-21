import torch

from train_structured import StructuredDataset


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 20) for char in text]


def test_target_labels_are_preserved_and_prompt_is_masked():
    rows = [{"prompt": "<TASK>\n", "target": "answer"}]
    dataset = StructuredDataset(rows, _Tokenizer(), max_length=32)
    ids, labels = dataset[0]
    assert all(value == -100 for value in labels[:len("<TASK>\n")])
    assert any(value != -100 for value in labels[len("<TASK>\n"):])


def test_long_targets_are_reported_not_silently_truncated():
    rows = [{"prompt": "p", "target": "x" * 40}]
    dataset = StructuredDataset(rows, _Tokenizer(), max_length=10)
    assert len(dataset) == 0
    assert dataset.dropped_long_targets == 1


def test_retained_task_counts_are_available_for_balanced_sampling():
    rows = [
        {"task": "STRUCTURE", "prompt": "p", "target": "x"},
        {"task": "GRAMMAR", "prompt": "p", "target": "x"},
        {"task": "GRAMMAR", "prompt": "p", "target": "x"},
    ]
    dataset = StructuredDataset(rows, _Tokenizer(), max_length=10)
    assert dataset.task_counts == {"STRUCTURE": 1, "GRAMMAR": 2}
