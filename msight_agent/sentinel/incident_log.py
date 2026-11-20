"""Durable incident records: one JSON line per incident in incidents.jsonl, plus a
markdown report per incident for people to read."""
import json
import time
from pathlib import Path
from typing import Optional

from msight_agent.shared.paths import LOG_DIR

INCIDENT_DIR = LOG_DIR / "incidents"
JSONL_PATH = INCIDENT_DIR / "incidents.jsonl"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def render_markdown(incident: dict, investigation: Optional[dict]) -> str:
    delay = incident["detected_at"] - incident["first_event_at"]
    lines = [
        f"# {incident['id']}: {incident['kind']} on {incident['node']}",
        "",
        f"- **Detected:** {_when(incident['detected_at'])} (first sign {_when(incident['first_event_at'])}, "
        f"{delay:.0f}s earlier)",
        f"- **What the sentinel saw:** {incident['summary']}",
        "",
        "## Timeline",
        "",
        *[f"- `{entry}`" for entry in incident["timeline"]],
        "",
    ]

    report = (investigation or {}).get("report")
    if report:
        lines += [
            "## Claude's diagnosis",
            "",
            f"**Summary:** {report.get('summary', '')}",
            "",
            f"**Root cause** ({report.get('category', '?')}, confidence {report.get('confidence', '?')}): "
            f"{report.get('root_cause', '')}",
            "",
            "**Evidence**",
            "",
            *[f"- {e}" for e in report.get("evidence", [])],
            "",
            f"**Affected:** {', '.join(report.get('affected', [])) or 'none'}",
            "",
        ]
        if report.get("code_references"):
            lines += ["**Code references**", ""]
            lines += [f"- `{r.get('path', '')}{':' + str(r['line']) if r.get('line') else ''}`: {r.get('why', '')}"
                      for r in report["code_references"]]
            lines.append("")
        lines += [
            "**Suggested fix (not applied):** " + report.get("suggested_fix", ""),
            "",
            "**Prevention:** " + report.get("prevention", ""),
            "",
        ]
    elif investigation:
        lines += ["## Claude's diagnosis", "", investigation.get("error") or investigation.get("text") or "(none)", ""]

    if investigation and investigation.get("checks"):
        lines += ["## Checks against measured facts", "",
                  *[f"- ⚠ {w}" for w in investigation["checks"]], ""]

    if investigation:
        lines += [
            "## Investigation",
            "",
            f"- Model: {investigation.get('model')}, {investigation.get('turns')} turns, "
            f"{investigation.get('duration_s', 0):.0f}s, ${investigation.get('cost_usd') or 0:.3f}",
            f"- Tool calls ({len(investigation.get('tool_calls', []))}):",
            *[f"  - `{call}`" for call in investigation.get("tool_calls", [])],
        ]
        if investigation.get("denied"):
            lines += ["- Denied by the read-only policy:", *[f"  - `{d}`" for d in investigation["denied"]]]
        lines.append("")

    lines += ["## Evidence collected by the sentinel", "", "```json",
              json.dumps(incident["evidence"], indent=2, default=str), "```", ""]
    return "\n".join(lines)


def append_event(incident_id: str, stage: str, **fields) -> None:
    """A later stage of an incident (resolved / dismissed) as its own line."""
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps({"id": incident_id, "stage": stage, "at": time.time(), **fields}, default=str) + "\n")


def write_incident(incident: dict, investigation: Optional[dict]) -> Path:
    """Append to incidents.jsonl and write the markdown report; returns the report path."""
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps({**incident, "investigation": investigation}, default=str) + "\n")
    report_path = INCIDENT_DIR / f"{incident['id']}.md"
    report_path.write_text(render_markdown(incident, investigation))
    return report_path
