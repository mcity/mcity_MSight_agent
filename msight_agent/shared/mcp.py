"""The one FastMCP instance every tool registers on, and the {"status": ...} response builders."""
import json

from fastmcp import FastMCP

mcp = FastMCP("MSight Agent")


def error_json(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def ok_json(message: str | None = None, **extra) -> str:
    result = {"status": "ok"}
    if message is not None:
        result["message"] = message
    result.update(extra)
    return json.dumps(result)
