import sys


def test_package_import_does_not_load_optional_mcp_server():
    import qcml_mcp

    assert "qcml_mcp.server" not in sys.modules
