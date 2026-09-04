import subprocess
import sys

import pytest


def test_package_import_does_not_load_optional_mcp_server():
    # Run in a fresh interpreter: pytest imports every test module during
    # collection, so sys.modules in-process is not a clean slate.
    script = (
        "import sys; import qcml_mcp; "
        "assert 'qcml_mcp.server' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# Hides the optional dependency even in environments that do have it installed,
# so this exercises the fallback path rather than silently passing.
_IMPORT_SERVER_WITHOUT_MCP = """
import sys


class _BlockMCP:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("No module named " + repr(fullname))
        return None


sys.meta_path.insert(0, _BlockMCP())

import qcml_mcp.server as server

assert server.MCP_AVAILABLE is False, "expected the optional dependency to look absent"
assert server.add(2, 3) == 5, "tool functions must stay callable without the dependency"

try:
    server.mcp.run()
except ImportError as exc:
    assert "qcmlforge[mcp]" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError from the stand-in server")
"""


def test_server_module_imports_without_the_optional_mcp_dependency():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SERVER_WITHOUT_MCP],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.mcp
def test_server_uses_real_fastmcp_when_the_dependency_is_installed():
    pytest.importorskip(
        "mcp.server.fastmcp",
        reason="requires the optional MCP server dependency",
    )
    from mcp.server.fastmcp import FastMCP

    import qcml_mcp.server as server

    assert server.MCP_AVAILABLE is True
    assert isinstance(server.mcp, FastMCP)
    assert server.add(2, 3) == 5
