"""Apply a bounded, auditable adjudication batch to the lexical task queue."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from xnlp_language.build_foundation import WORD


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = ROOT / "xnlp_language" / "generated" / "annotation" / "lexical_analysis_queue.jsonl"
DEFAULT_REPORT = ROOT / "xnlp_language" / "generated" / "reports" / "lexical_adjudication_progress.json"
DEFAULT_CORPUS = ROOT / "xnlp_language" / "generated" / "evidence" / "corpus_records.jsonl"
DEFAULT_REGISTRY = ROOT / "xnlp_language" / "generated" / "evidence" / "source_provenance.json"
ASSIGNED_FIELDS = ("assigned_lemma", "assigned_category", "assigned_meaning")
PROPER_NOUN_RULE = "AUTO_PROPER_NOUN_CAPITALIZED_MULTIWORK_V1"


def _decode_line(line: bytes) -> tuple[dict[str, Any], bytes]:
    if line.endswith(b"\r\n"):
        return json.loads(line[:-2]), b"\r\n"
    if line.endswith(b"\n") or line.endswith(b"\r"):
        return json.loads(line[:-1]), line[-1:]
    return json.loads(line), b""


def _read_adjudications(path: Path) -> dict[str, dict[str, Any]]:
    adjudications = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Adjudication line {line_number} requires a task_id")
        if task_id in adjudications:
            raise ValueError(f"Duplicate adjudication for task {task_id}")
        adjudications[task_id] = row
    return adjudications


def _validate_adjudication(
    task_id: str,
    adjudication: dict[str, Any],
    machine_assisted: bool,
) -> dict[str, Any]:
    status = adjudication.get("status")
    adjudicator = adjudication.get("adjudicator")
    if status not in {"adjudicated", "disputed"}:
        raise ValueError(f"Task {task_id} status must be adjudicated or disputed")
    flagged_by = adjudication.get("flagged_by")
    valid_adjudicator = adjudicator in {"human", "rule-assisted-human", "machine-only"}
    valid_machine_flag = (
        status == "disputed" and adjudicator is None and flagged_by == "machine-only"
    )
    if not valid_adjudicator and not valid_machine_flag:
        raise ValueError(f"Task {task_id} requires a recognized adjudicator")
    if adjudicator == "machine-only" and not machine_assisted:
        raise ValueError(
            f"Task {task_id} is machine-only; pass --machine-assisted to permit this write"
        )
    note = adjudication.get("adjudication_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"Task {task_id} requires a non-empty adjudication_note")
    normalized = {
        "status": status,
        "assigned_lemma": adjudication.get("assigned_lemma"),
        "assigned_category": adjudication.get("assigned_category"),
        "assigned_meaning": adjudication.get("assigned_meaning"),
        "adjudicator": None if status == "disputed" and adjudicator == "machine-only" else adjudicator,
        "adjudication_note": note.strip(),
        "flagged_by": (
            "machine-only"
            if status == "disputed" and (
                adjudicator == "machine-only" or flagged_by == "machine-only"
            )
            else None
        ),
        "auto_evidence": None,
        "adjudication_history": [],
    }
    for field in ASSIGNED_FIELDS:
        if normalized[field] is not None and not isinstance(normalized[field], str):
            raise ValueError(f"Task {task_id} {field} must be a string or null")
    if status == "disputed" and any(normalized[field] is not None for field in ASSIGNED_FIELDS):
        raise ValueError(f"Disputed task {task_id} must not assign lemma, category, or meaning")
    if status == "adjudicated" and not any(normalized[field] for field in ASSIGNED_FIELDS):
        raise ValueError(f"Adjudicated task {task_id} requires at least one assigned value")
    return normalized


def apply_batch(
    queue_path: Path,
    limit: int,
    adjudications: dict[str, dict[str, Any]] | None = None,
    *,
    machine_assisted: bool = False,
    default_disputed: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("--limit must be greater than zero")
    if default_disputed and adjudications:
        raise ValueError("--default-disputed cannot be combined with imported adjudications")
    original = queue_path.read_bytes()
    raw_lines = original.splitlines(keepends=True)
    parsed = [_decode_line(line)[0] for line in raw_lines if line.strip()]
    if len(parsed) < limit:
        raise ValueError(f"Queue contains {len(parsed)} tasks, fewer than requested batch size {limit}")
    batch = parsed[:limit]
    if any(task.get("status", "pending") != "pending" for task in batch):
        raise ValueError("The current front-of-queue batch contains non-pending tasks")
    batch_ids = [task.get("task_id") for task in batch]
    if any(not isinstance(task_id, str) or not task_id for task_id in batch_ids):
        raise ValueError("Every task in the current batch must have a task_id")

    if default_disputed:
        imported = {
            task_id: {
                "status": "disputed",
                "assigned_lemma": None,
                "assigned_category": None,
                "assigned_meaning": None,
                "adjudicator": None,
                "adjudication_note": "requires human isiXhosa review",
                "flagged_by": "machine-only",
                "auto_evidence": None,
                "adjudication_history": [],
            }
            for task_id in batch_ids
        }
    else:
        imported = adjudications or {}
        unknown_ids = sorted(set(imported) - set(batch_ids))
        missing_ids = sorted(set(batch_ids) - set(imported))
        if unknown_ids or missing_ids:
            raise ValueError(
                f"Adjudications must match the current batch exactly; "
                f"missing={missing_ids[:5]}, outside_batch={unknown_ids[:5]}"
            )

    updated_by_id = {
        task_id: _validate_adjudication(task_id, imported[task_id], machine_assisted)
        for task_id in batch_ids
    }
    updated_lines: list[bytes] = []
    task_index = 0
    for raw_line in raw_lines:
        if not raw_line.strip():
            updated_lines.append(raw_line)
            continue
        task = parsed[task_index]
        if task_index < limit:
            for field, value in updated_by_id[task["task_id"]].items():
                task[field] = value
            _, line_ending = _decode_line(raw_line)
            updated_lines.append(
                json.dumps(task, ensure_ascii=False, sort_keys=True).encode("utf-8")
                + line_ending
            )
        else:
            updated_lines.append(raw_line)
        task_index += 1

    result_bytes = b"".join(updated_lines)
    queue_path.write_bytes(result_bytes)
    final_rows = [_decode_line(line)[0] for line in result_bytes.splitlines(keepends=True) if line.strip()]
    totals = Counter(row.get("status", "pending") for row in final_rows)
    tasks_requiring_human_attention = sum(
        row.get("status") in {"pending", "disputed"}
        or any(row.get(field) is None for field in ASSIGNED_FIELDS)
        for row in final_rows
    )
    return {
        "queue_path": str(queue_path),
        "queue_tasks": len(final_rows),
        "batch_size": limit,
        "batch_task_ids": batch_ids,
        "batch_status_counts": dict(sorted(Counter(
            row["status"] for row in final_rows[:limit]
        ).items())),
        "overall_status_counts": {
            status: totals[status] for status in ("pending", "adjudicated", "disputed")
        },
        "tasks_requiring_human_attention": tasks_requiring_human_attention,
        "tasks_with_unassigned_fields": sum(
            any(row.get(field) is None for field in ASSIGNED_FIELDS)
            for row in final_rows
        ),
        "machine_assisted_write": any(
            row["adjudicator"] == "machine-only" for row in updated_by_id.values()
        ),
        "flagged_by_machine_only": sum(
            row.get("flagged_by") == "machine-only" for row in final_rows
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _work_reference(record: dict[str, Any], registry_sources: dict[str, Any]) -> str | None:
    attribution = record.get("work_attribution")
    if isinstance(attribution, dict) and isinstance(attribution.get("work"), str):
        return attribution["work"]
    source = registry_sources.get(str(record.get("source_title", "")), {})
    group = source.get("independence_group")
    return group if isinstance(group, str) and group else None


def propose_auto_adjudications(
    tasks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    registry_sources: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source_title = str(record.get("source_title", ""))
        source = registry_sources.get(source_title)
        provenance_type = str(record.get("provenance_type", "")).casefold()
        if (
            not source
            or source.get("foundation_attested") is not True
            or provenance_type != "authoritative"
            or record.get("generator")
        ):
            continue
        work = _work_reference(record, registry_sources)
        if not work:
            continue
        text = str(record.get("text", ""))
        for match in WORD.finditer(text):
            surface = match.group(0)
            sentence_start = max(
                text.rfind(mark, 0, match.start())
                for mark in (".", "!", "?", "\n")
            )
            prior_text = text[sentence_start + 1:match.start()]
            mid_sentence = any(char.isalpha() for char in prior_text)
            occurrences.setdefault(surface.casefold(), []).append({
                "record_id": str(record["record_id"]),
                "work": work,
                "capitalized": surface[:1].isupper(),
                "mid_sentence": mid_sentence,
            })

    proposals = {}
    for task in tasks:
        if task.get("status", "pending") != "pending" or task.get("flagged_by"):
            continue
        form = str(task.get("surface_form", "")).casefold()
        evidence = occurrences.get(form, [])
        work_counts = Counter(
            hit["work"] for hit in evidence if hit["capitalized"] and hit["mid_sentence"]
        )
        qualifying_works = sorted(work for work, count in work_counts.items() if count >= 2)
        if (
            len(qualifying_works) < 2
            or not evidence
            or any(not hit["capitalized"] for hit in evidence)
        ):
            continue
        supporting = [
            hit for hit in evidence
            if hit["work"] in qualifying_works and hit["capitalized"] and hit["mid_sentence"]
        ]
        record_ids = sorted({hit["record_id"] for hit in supporting})
        proposals[str(task["task_id"])] = {
            "status": "adjudicated",
            "assigned_lemma": None,
            "assigned_category": "proper_noun",
            "assigned_meaning": None,
            "adjudicator": "auto-evidence",
            "adjudication_note": (
                f"{PROPER_NOUN_RULE}; supporting record IDs: {', '.join(record_ids)}"
            ),
            "flagged_by": None,
            "auto_evidence": {
                "rule_ids": [PROPER_NOUN_RULE],
                "supporting_record_ids": record_ids,
                "supporting_work_references": qualifying_works,
            },
        }
    return proposals


def _write_task_changes(
    queue_path: Path,
    changes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    original = queue_path.read_bytes()
    raw_lines = original.splitlines(keepends=True)
    final_rows = []
    updated_lines = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            updated_lines.append(raw_line)
            continue
        task, line_ending = _decode_line(raw_line)
        update = changes.get(str(task.get("task_id")))
        if update:
            for field, value in update.items():
                task[field] = value
            raw_line = (
                json.dumps(task, ensure_ascii=False, sort_keys=True).encode("utf-8")
                + line_ending
            )
        updated_lines.append(raw_line)
        final_rows.append(task)
    if changes:
        queue_path.write_bytes(b"".join(updated_lines))
    return final_rows


def bootstrap_queue(
    queue_path: Path = DEFAULT_QUEUE,
    corpus_path: Path = DEFAULT_CORPUS,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    tasks = _load_jsonl(queue_path)
    records = _load_jsonl(corpus_path)
    registry_sources = json.loads(registry_path.read_text(encoding="utf-8")).get("sources", {})
    proposals = propose_auto_adjudications(tasks, records, registry_sources)
    changes = {}
    for task_id, proposal in proposals.items():
        current = next(task for task in tasks if task.get("task_id") == task_id)
        history = list(current.get("adjudication_history") or [])
        history.append({
            "action": "auto-adjudicated",
            "adjudicator": "auto-evidence",
            "assigned_lemma": proposal["assigned_lemma"],
            "assigned_category": proposal["assigned_category"],
            "assigned_meaning": proposal["assigned_meaning"],
            "note": proposal["adjudication_note"],
            "rule_ids": proposal["auto_evidence"]["rule_ids"],
            "supporting_record_ids": proposal["auto_evidence"]["supporting_record_ids"],
        })
        changes[task_id] = {**proposal, "adjudication_history": history}
    final_rows = _write_task_changes(queue_path, changes)
    totals = Counter(row.get("status", "pending") for row in final_rows)
    tasks_requiring_human_attention = sum(
        row.get("status") in {"pending", "disputed"}
        or any(row.get(field) is None for field in ASSIGNED_FIELDS)
        for row in final_rows
    )
    histogram = Counter(
        rule_id
        for row in final_rows
        if row.get("adjudicator") == "auto-evidence"
        for rule_id in (row.get("auto_evidence") or {}).get("rule_ids", [])
    )
    return {
        "queue_path": str(queue_path),
        "queue_tasks": len(final_rows),
        "adjudicated": totals["adjudicated"],
        "pending": totals["pending"],
        "disputed": totals["disputed"],
        "tasks_requiring_human_attention": tasks_requiring_human_attention,
        "tasks_with_unassigned_fields": sum(
            any(row.get(field) is None for field in ASSIGNED_FIELDS)
            for row in final_rows
        ),
        "auto_adjudicated_this_run": len(changes),
        "rule_firing_histogram": dict(sorted(histogram.items())),
        "rule_definition": {
            PROPER_NOUN_RULE: (
                "A form must occur only capitalized in attested records and have at least "
                "two mid-sentence capitalized occurrences in each of at least two independent "
                "work references. No lemma or meaning is inferred."
            )
        },
    }


def revert_batch(
    queue_path: Path = DEFAULT_QUEUE,
    limit: int = 1,
    reverted_by: str = "human-reviewer",
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("--revert-batch must be greater than zero")
    if not reverted_by.strip():
        raise ValueError("--reverted-by must not be empty")
    tasks = _load_jsonl(queue_path)
    selected = [
        row for row in tasks if row.get("adjudicator") == "auto-evidence"
    ][:limit]
    if len(selected) != limit:
        raise ValueError(f"Queue contains only {len(selected)} auto-evidence decisions to revert")
    changes = {}
    for row in selected:
        auto_evidence = row.get("auto_evidence") or {}
        history = list(row.get("adjudication_history") or [])
        history.append({
            "action": "human-reverted",
            "adjudicator": "auto-evidence",
            "assigned_lemma": row.get("assigned_lemma"),
            "assigned_category": row.get("assigned_category"),
            "assigned_meaning": row.get("assigned_meaning"),
            "note": f"Reverted by {reverted_by.strip()} to pending; retained auto-evidence audit.",
            "rule_ids": auto_evidence.get("rule_ids", []),
            "supporting_record_ids": auto_evidence.get("supporting_record_ids", []),
        })
        changes[row["task_id"]] = {
            "status": "pending",
            "assigned_lemma": None,
            "assigned_category": None,
            "assigned_meaning": None,
            "adjudicator": None,
            "adjudication_note": None,
            "adjudication_history": history,
        }
    final_rows = _write_task_changes(queue_path, changes)
    totals = Counter(row.get("status", "pending") for row in final_rows)
    return {
        "queue_path": str(queue_path),
        "reverted_tasks": len(changes),
        "reverted_task_ids": [row["task_id"] for row in selected],
        "overall_status_counts": {
            status: totals[status] for status in ("pending", "adjudicated", "disputed")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--adjudications",
        type=Path,
        help="JSONL import with one adjudication object per task in the current batch.",
    )
    parser.add_argument(
        "--default-disputed",
        action="store_true",
        help="Mark the current batch disputed without assigning linguistic content.",
    )
    parser.add_argument(
        "--machine-assisted",
        action="store_true",
        help="Explicitly permit machine-only adjudication records.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Apply conservative evidence-only decisions across the full queue.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Corpus evidence JSONL used by --bootstrap.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Source provenance registry JSON used by --bootstrap.",
    )
    parser.add_argument(
        "--revert-batch",
        type=int,
        help="Revert the first N auto-evidence decisions to pending, retaining history.",
    )
    parser.add_argument("--reverted-by", default="human-reviewer")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    selected_modes = sum((
        args.bootstrap,
        args.revert_batch is not None,
        args.adjudications is not None,
        args.default_disputed,
    ))
    if selected_modes != 1:
        parser.error("choose exactly one of --bootstrap, --revert-batch, --adjudications, or --default-disputed")
    if args.default_disputed and args.adjudications:
        parser.error("--default-disputed and --adjudications are mutually exclusive")
    if args.bootstrap:
        progress = bootstrap_queue(args.queue, args.corpus, args.registry)
    elif args.revert_batch is not None:
        progress = revert_batch(args.queue, args.revert_batch, args.reverted_by)
    else:
        if args.limit is None:
            parser.error("--limit is required for batch adjudication")
        imported = _read_adjudications(args.adjudications) if args.adjudications else None
        progress = apply_batch(
            args.queue,
            args.limit,
            imported,
            machine_assisted=args.machine_assisted,
            default_disputed=args.default_disputed,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Adjudication failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
