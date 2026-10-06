# Tool modules are imported only to register their @mcp.tool() functions.
from mcptools import (  # noqa: F401
    mcp,
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
)

if __name__ == "__main__":
    # access_log=False: each SSE JSON-RPC message is its own POST, flooding the log.
    mcp.run(transport="sse", host="0.0.0.0", port=8000, uvicorn_config={"access_log": False})
