"""MCP tool: is real (non-default) camera calibration in place?"""
from msight_agent.msight.runtime.calibration import calibration_message, calibration_status
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.shared.mcp import error_json, mcp, ok_json


@mcp.tool()
def check_msight_calibration_status() -> str:
    msight_path, err = get_msight_path()
    if err:
        return error_json(err)
    status = calibration_status(msight_path)
    return ok_json(calibration_message(status["state"]), **status)
