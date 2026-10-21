import json
from tempfile import TemporaryDirectory
from pathlib import Path

import train_prose_pilot
from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.data import load_corpus, prepare_data, tokenize_text_streams


def test_pilot_stages_only_train_and_validation(monkeypatch):
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        train = root / "train.txt"
        validation = root / "val.txt"
        test = root / "test_provenance.jsonl"
        train.write_text("train sentence\n", encoding="utf-8")
        validation.write_text("validation sentence\n", encoding="utf-8")
        test.write_text('{"text":"held-out test text"}\n', encoding="utf-8")
        monkeypatch.setattr(train_prose_pilot, "DEV_FILES", (train, validation))
        monkeypatch.setattr(train_prose_pilot, "OUTER_TEST", test)

        result = train_prose_pilot.prepare_development_corpus(root / "staged")
        staged_text = Path(result["staged_corpus"]).read_text(encoding="utf-8")

        assert staged_text == "train sentence\nvalidation sentence\n"
        assert "held-out test text" not in staged_text
        assert result["outer_test_provenance_sha256"]


def test_pilot_corpus_retains_line_break_tokens():
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        (root / "development.txt").write_text(
            "umgca wokuqala\n\numgca wokuqala\n", encoding="utf-8"
        )

        sentences = load_corpus(str(root), preserve_line_breaks=True)
        tokenizer = XNLPTokenizer(vocab_size=64, min_frequency=1)
        tokenizer.train(sentences, verbose=False)

        assert sentences == ["umgca wokuqala\n", "\n", "umgca wokuqala\n"]
        for sentence in sentences:
            encoded = tokenizer.encode(sentence, add_special_tokens=False)
            assert tokenizer.decode(encoded) == sentence


def test_pilot_packs_multiple_rows_without_intermediate_end_tokens():
    sentences = ["umgca wokuqala\n", "umgca wesibini\n", "umgca wesithathu\n"]
    tokenizer = XNLPTokenizer(vocab_size=64, min_frequency=1)
    tokenizer.train(sentences, verbose=False)

    sequences = tokenize_text_streams(sentences, tokenizer, max_len=12)

    assert len(sequences) > 1
    assert all(len(sequence) <= 12 for sequence in sequences)
    assert all(sequence[0] == tokenizer.bos_token_id for sequence in sequences)
    assert all(tokenizer.eos_token_id not in sequence for sequence in sequences[:-1])
    assert sequences[-1][-1] == tokenizer.eos_token_id
    decoded = "".join(
        tokenizer.decode(sequence, skip_special_tokens=True)
        for sequence in sequences
    )
    assert decoded == "".join(sentences)


def test_work_isolated_corpus_keeps_works_in_separate_predefined_splits():
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        source = root / "source.jsonl"
        source_records = [
            {
                "record_id": f"record-{index}",
                "text": f"Isivakalisi esikhethekileyo {index}.",
                "source": "mqhayi",
                "source_title": title,
                "language_class": "TARGET_LANGUAGE",
                "validation_status": "authoritative",
            }
            for index, title in enumerate(("train work", "validation work", "test work"))
        ]
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_records),
            encoding="utf-8",
        )

        result = train_prose_pilot.prepare_work_isolated_corpus(
            root / "grouped",
            source_provenance=source,
            validation_work="validation work",
            test_work="test work",
        )

        assert result["splits"]["train"]["works"] == ["train work"]
        assert result["splits"]["val"]["works"] == ["validation work"]
        assert result["splits"]["test"]["works"] == ["test work"]
        assert Path(result["test_provenance"]).is_file()
        assert Path(result["splits"]["val"]["provenance_path"]).is_file()


def test_predefined_split_loader_trains_tokenizer_only_on_training_partition():
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        for split, text in (
            ("train", "uloliwe uqhubeka phambili\nisixhosa lulwimi lwethu\n"),
            ("val", "abantu basebenza kunye\n"),
            ("test", "sifunda ulwimi lwethu\n"),
        ):
            split_dir = root / split
            split_dir.mkdir()
            (split_dir / f"{split}.txt").write_text(text, encoding="utf-8")

        config = TrainingConfig(
            corpus_dir=str(root),
            preset="tiny",
            vocab_size=64,
            tokenizer_min_freq=1,
            max_seq_len=32,
            predefined_splits=True,
            preserve_line_breaks=True,
            device="cpu",
        )
        train_loader, val_loader, test_loader, tokenizer = prepare_data(
            config, verbose=False, include_test=True
        )

        assert tokenizer.training_data_size == 2
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0
