# Tool modules are imported only to register their @mcp.tool() functions.
from mcp_layer.mcptools import (  # noqa: F401
    mcp,
    workflow_selector,
)
from mcp_layer.mcptools.auto_labeling import (  # noqa: F401
    auto_labeling,
    cvat_export,
    data_ingest,
    label_studio_export,
    v51,
)
from mcp_layer.mcptools.msight import (  # noqa: F401
    docker,
    host_inspect,
    nodes,
    record_archive,
    reference,
)

if __name__ == "__main__":
    # access_log=False: each SSE JSON-RPC message is its own POST, flooding the log.
    mcp.run(transport="sse", host="0.0.0.0", port=8000, uvicorn_config={"access_log": False})
