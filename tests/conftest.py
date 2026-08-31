import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_PRETRAINED_MODEL_GROUPS = {
    "am": ["am_ensemble/am_0.pt"],
    "am_ensemble": [f"am_ensemble/am_{index}.pt" for index in range(5)],
    "ap2": ["am_ensemble/am_0.pt", "ap2_ensemble/ap2_0.pt"],
    "ap2_ensemble": [
        *[f"am_ensemble/am_{index}.pt" for index in range(5)],
        *[f"ap2_ensemble/ap2_{index}.pt" for index in range(5)],
    ],
    "ap2_fused_ensemble": [
        "ap2-fused_ensemble/ap2_1.pt",
        "ap2-fused_ensemble/ap2_2.pt",
        "ap2-fused_ensemble/ap2_3.pt",
    ],
    "dapnet2": [
        "dapnet2/backbone/am_0.pt",
        "dapnet2/backbone/ap2_0.pt",
        "dapnet2/B3LYP-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    ],
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y"}


@pytest.fixture(autouse=True)
def require_pretrained_models(request):
    """Skip marked tests unless all requested pretrained artifacts resolve."""
    marker = request.node.get_closest_marker("pretrained_models")
    if marker is None:
        return

    env_value = os.getenv("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "").strip().lower()
    if env_value not in _TRUE_ENV_VALUES:
        pytest.skip(
            "pretrained model downloads are disabled; set "
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1"
        )

    unknown_groups = set(marker.args) - _PRETRAINED_MODEL_GROUPS.keys()
    if unknown_groups:
        raise ValueError(
            f"Unknown pretrained model groups: {sorted(unknown_groups)}"
        )

    rel_paths = [
        rel_path
        for group in marker.args
        for rel_path in _PRETRAINED_MODEL_GROUPS[group]
    ]
    resolved_paths = []
    if rel_paths:
        try:
            from apnet_pt.hf_pretrained import resolve_pretrained_paths

            resolved = resolve_pretrained_paths(list(dict.fromkeys(rel_paths)))
            resolved_paths.extend(resolved.values())
        except (ImportError, OSError, RuntimeError) as exc:
            pytest.skip(f"required pretrained models could not be resolved: {exc}")

    resolved_paths.extend(ROOT / path for path in marker.kwargs.get("local", []))
    missing_paths = [str(path) for path in resolved_paths if not Path(path).is_file()]
    if missing_paths:
        pytest.skip(
            "required pretrained model paths do not exist: "
            + ", ".join(missing_paths)
        )
