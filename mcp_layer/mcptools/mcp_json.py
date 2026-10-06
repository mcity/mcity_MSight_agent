"""Builders for the {"status": "ok"/"error", ...} MCP tool response shape."""
import json


def error_json(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def ok_json(message: str | None = None, **extra) -> str:
    result = {"status": "ok"}
    if message is not None:
        result["message"] = message
    result.update(extra)
    return json.dumps(result)
