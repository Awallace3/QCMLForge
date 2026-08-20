import subprocess
import sys


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
