"""Routes MCP log notifications to the in-flight /chat/stream request (single-flight)."""

_active_progress_cb = None


def set_active_progress_cb(cb):
    global _active_progress_cb
    _active_progress_cb = cb


def clear_active_progress_cb():
    global _active_progress_cb
    _active_progress_cb = None


def get_active_progress_cb():
    return _active_progress_cb
