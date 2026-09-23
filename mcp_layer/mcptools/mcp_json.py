"""Shared response-JSON builders for MCP tool functions -- factors out the
{"status": "ok"/"error", ...} shape repeated across mcptools/*.py."""
import json


def error_json(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def ok_json(message: str | None = None, **extra) -> str:
    result = {"status": "ok"}
    if message is not None:
        result["message"] = message
    result.update(extra)
    return json.dumps(result)
