from . import timings

__all__ = ["server", "timings"]


def __getattr__(name):
    # `server` pulls in the optional `mcp` SDK. Import it lazily so that
    # `qcml_mcp.ie_time_esimator_script` and `qcml_mcp.timings` remain usable
    # in environments without the MCP server dependencies installed.
    if name == "server":
        from . import server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
