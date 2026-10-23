"""MCP tool server (port 8000). Tool modules are imported only to register their
@mcp.tool() functions on the shared instance."""
from msight_agent.agent import workflow_tools  # noqa: F401
from msight_agent.auto_labeling.tools import (  # noqa: F401
    cvat,
    data_ingest,
    label_studio,
    labeling,
    voxel51,
)
from msight_agent.msight.tools import (  # noqa: F401
    calibration,
    host_inspect,
    nodes,
    pipeline,
    record_archive,
    reference,
)
from msight_agent.shared.mcp import mcp

if __name__ == "__main__":
    # access_log=False: each SSE JSON-RPC message is its own POST, flooding the log.
    mcp.run(transport="sse", host="0.0.0.0", port=8000, uvicorn_config={"access_log": False})
