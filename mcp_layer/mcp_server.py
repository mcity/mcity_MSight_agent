from mcptools import (
    mcp,  #shared FastMCP instance from __init__.py
    workflow_selector,
    auto_labeling,
    data_ingest,
    v51,
    cvat_export,
    label_studio_export,
    msight_docker,
    msight_record_archive,
    msight_nodes,
    msight_reference,
    # msight_calibration_helper intentionally not imported -- auto-detect-intrinsics is benched.
)

if __name__ == "__main__":
    # access_log=False: SSE is one-directional, so every JSON-RPC message
    # (each tool call, each response) travels as its own POST /messages/
    # request on a side channel -- uvicorn's default per-request access log
    # turns into hundreds of "202 Accepted" lines within an hour of normal
    # polling, burying real logging.warning/error output in the noise.
    mcp.run(transport="sse", host="0.0.0.0", port=8000, uvicorn_config={"access_log": False})
