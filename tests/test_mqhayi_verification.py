from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

from evaluate_mqhayi_checkpoint import (
    DEFAULT_CHECKPOINT,
    DEFAULT_BASELINE_TRAINING_TEXT,
    DEFAULT_TEST_PROVENANCE,
    evaluate_mqhayi_checkpoint,
    select_mqhayi_record_ids,
)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="baseline checkpoint not available")
def test_baseline_checkpoint_on_non_overlapping_mqhayi_test_records():
    record_ids = select_mqhayi_record_ids(
        DEFAULT_TEST_PROVENANCE,
        (DEFAULT_BASELINE_TRAINING_TEXT,),
        sample_size=8,
    )
    report = evaluate_mqhayi_checkpoint(
        DEFAULT_CHECKPOINT,
        sample_size=8,
        test_provenance=DEFAULT_TEST_PROVENANCE,
        record_ids=record_ids,
    )

    assert report["selection"]["source"] == "mqhayi"
    assert report["selection"]["test_records"] > 0
    assert report["selection"]["eligible_non_overlapping_records"] > 0
    assert report["selection"]["sample_size"] == 8
    assert report["selection"]["sample_record_ids"] == record_ids
    assert report["selection"]["exact_training_text_overlaps_excluded"] > 0
    assert report["tokenizer_round_trip"]["total"] == 8
    assert (
        report["tokenizer_round_trip"]["whitespace_only_failure_count"]
        + report["tokenizer_round_trip"]["other_surface_loss_count"]
        + report["tokenizer_round_trip"]["passed"]
        == 8
    )
    assert report["next_token_evaluation"]["target_tokens"] > 0
    assert report["next_token_evaluation"]["cross_entropy_loss"] > 0
    assert report["next_token_evaluation"]["perplexity"] > 1
    assert "does not verify semantics" in report["interpretation"]


def test_shared_sample_excludes_records_in_either_training_corpus():
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        test_provenance = root / "test.jsonl"
        test_provenance.write_text(
            "\n".join(
                [
                    '{"source":"mqhayi","language_class":"TARGET_LANGUAGE","text":"shared text","record_id":"shared"}',
                    '{"source":"mqhayi","language_class":"TARGET_LANGUAGE","text":"baseline only","record_id":"baseline"}',
                    '{"source":"mqhayi","language_class":"TARGET_LANGUAGE","text":"pilot only","record_id":"pilot"}',
                    '{"source":"mqhayi","language_class":"TARGET_LANGUAGE","text":"eligible text","record_id":"eligible"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        baseline = root / "baseline.txt"
        pilot = root / "pilot.txt"
        baseline.write_text("baseline only\n", encoding="utf-8")
        pilot.write_text("pilot only\n", encoding="utf-8")

        ids = select_mqhayi_record_ids(test_provenance, (baseline, pilot), sample_size=10)

        assert ids == ["shared", "eligible"]
