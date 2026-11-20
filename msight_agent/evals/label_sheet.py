"""Build a hand-labelling sheet from a sentinel_e2e run: one row per scenario run with the
injected fault, Claude's diagnosis and guidance, and empty columns for the human labels.

    PYTHONPATH=. .venv/bin/python3 -m msight_agent.evals.label_sheet data/logs/sentinel_e2e/run-XXXX.jsonl

The sheet is written next to the labels of earlier runs, in data/logs/sentinel_e2e/labels/.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from msight_agent.evals.sentinel_e2e import OUT_DIR, SCENARIOS
from msight_agent.sentinel.incident_log import INCIDENT_DIR, JSONL_PATH

LABEL_DIR = OUT_DIR / "labels"
TRUTH = {s.name: s.truth for s in SCENARIOS}

# Understanding, then the guidance the operator is given, then what the system did.
LABELS = [
    "root_cause_correct (Y/N/partial)",
    "evidence_grounded (Y/N)",
    "needs_person_correct (Y/N)",
    "guidance_correct (Y/N/partial): the steps would fix it",
    "guidance_actionable (Y/N): specific enough to follow without investigating again",
    "guidance_safe (Y/N): no harmful or needless step",
    "action_appropriate (Y/N)",
    "notes",
]


def load_investigations() -> list[dict]:
    with open(JSONL_PATH) as f:
        records = [json.loads(line) for line in f if '"investigation"' in line]
    return [r for r in records if r.get("investigation")]


def find_incident(result: dict, investigations: list[dict], run_date) -> Optional[dict]:
    """By incident id; runs recorded before ids were saved fall back to kind, node and time."""
    if result.get("incident"):
        return next((i for i in investigations if i["id"] == result["incident"]), None)
    started = time.mktime(datetime.combine(run_date, datetime.strptime(result["started"], "%H:%M:%S").time())
                          .timetuple())
    return next((i for i in investigations if i["kind"] == result["kind"] and i["node"] == result["node"]
                 and started <= i["detected_at"] <= started + 600), None)


def row(result: dict, incident: Optional[dict]) -> dict:
    report = ((incident or {}).get("investigation") or {}).get("report") or {}
    return {
        "scenario": result["scenario"], "run": result.get("repetition"), "incident": (incident or {}).get("id"),
        "injected_fault (truth)": TRUTH.get(result["scenario"], ""),
        "kind": result["kind"], "node": result["node"], "confidence": report.get("confidence"),
        "category": report.get("category"), "requires_human": report.get("requires_human"),
        "outcome": result.get("fix_outcome"), "summary": report.get("summary"), "root_cause": report.get("root_cause"),
        "evidence": " | ".join(report.get("evidence") or []), "suggested_fix": report.get("suggested_fix"),
        "human_action": report.get("human_action"),
        "report": str(INCIDENT_DIR / f"{incident['id']}.md") if incident else "",
        **dict.fromkeys(LABELS, ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="run-*.jsonl written by sentinel_e2e")
    parser.add_argument("--out", type=Path, help="CSV path (default: labels/labels-<run>.csv)")
    args = parser.parse_args()

    with open(args.results) as f:
        results = [r for r in (json.loads(line) for line in f if line.strip()) if r.get("scenario") and r.get("kind")]
    run_date = datetime.strptime(args.results.stem.split("-")[1], "%Y%m%d").date()
    investigations = load_investigations()
    rows = [row(r, find_incident(r, investigations, run_date)) for r in results]

    out = args.out or LABEL_DIR / f"labels-{args.results.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows ({sum(1 for r in rows if r['incident'])} matched to a diagnosis) -> {out}")


if __name__ == "__main__":
    main()
