from mcptools import (
    mcp,  #shared FastMCP instance from __init__.py
    workflow_selector,
    auto_labeling,
    data_ingest,
    v51,
    cvat_export,
    label_studio_export
)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
