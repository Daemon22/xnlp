import json

from xnlp_trainer.structured_curriculum import TASKS, build_curriculum


def test_curriculum_is_provenance_linked_and_covers_all_tasks(tmp_path):
    source = tmp_path / "train.txt"
    source.write_text("Umntu uyafunda isiXhosa.\nUbuntu buhle.\n", encoding="utf-8")
    output = tmp_path / "curriculum.jsonl"
    report = build_curriculum(source, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["examples"] == len(rows)
    assert set(TASKS).issubset({row["task"] for row in rows})
    assert all(row["source_record_id"] and row["provenance"] for row in rows)
    assert all(row["dataset_version"] == "structured-curriculum-v2" for row in rows)
    assert all(row["annotation_status"] == "NOT_EXPERT_REVIEWED" for row in rows)
    assert all(isinstance(row["provenance"], dict) for row in rows)

    structure = next(row for row in rows if row["task"] == "STRUCTURE")
    assert "semantic_frames" not in json.loads(structure["target"])
    assert "schema_version" in json.loads(structure["target"])
    assert all(row["prompt"].startswith("<TASK:") for row in rows)
