"""Portable S66 viewer data preparation, independent of model dependencies."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/build_mastiff_spherical_vis.py"
SPEC = importlib.util.spec_from_file_location("spherical_vis", SCRIPT)
vis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vis)


def test_water_plan_and_fragment_isolation():
    symbols = ["O", "H", "H"]
    xyz = np.array([[0., 0., 0.], [.76, .59, 0.], [-.76, .59, 0.]])
    bonds, plans = vis.monomer_plan(symbols, xyz)
    assert bonds == [[0, 1], [0, 2]]
    assert plans[0] == {"kind": "full", "refs": [[1, 2], [2, 1]]}
    assert plans[1] == {"kind": "axial", "refs": [[0]]}
    perm = np.array([2, 0, 1])
    _, permuted = vis.monomer_plan([symbols[i] for i in perm], xyz[perm])
    remapped = {tuple(perm[r] for r in refs) for refs in permuted[1]["refs"]}
    assert remapped == {(1, 2), (2, 1)}


def test_rotation_does_not_change_reference_plan():
    xyz = np.array([[0., 0., 0.], [.76, .59, 0.], [-.76, .59, 0.]])
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    assert vis.monomer_plan(["O", "H", "H"], xyz) == vis.monomer_plan(
        ["O", "H", "H"], xyz @ q + 4
    )


def test_collinear_and_unbound_sites_are_explicit():
    _, plans = vis.monomer_plan(
        ["C", "H", "H", "O"],
        np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [8., 0., 0.]]),
    )
    assert plans[0]["kind"] == "axial"
    assert plans[3] == {"kind": "isotropic", "refs": []}


def test_parser_rejects_incomplete_database():
    import pytest

    with pytest.raises(ValueError, match="528"):
        vis.parse_database("")


def test_embedded_database_and_all_frame_geometries():
    html = (SCRIPT.parents[1] / "docs/mastiff-spherical-vis.html").read_text()
    data = json.loads(re.search(
        r'<script id="dataset" type="application/json">(.*?)</script>', html, re.S
    )[1])
    assert len(data["dimers"]) == 66
    for dimer in data["dimers"]:
        assert set(dimer["geometries"]) == set(vis.DISTANCES)
        split = dimer["split"]
        assert all((a < split) == (b < split) for a, b in dimer["bonds"])
        for coords in dimer["geometries"].values():
            xyz = np.array(coords)
            assert xyz.shape == (len(dimer["symbols"]), 3)
            assert np.isfinite(xyz).all()
            for i, plan in enumerate(dimer["plans"]):
                for refs in plan["refs"]:
                    assert all((i < split) == (ref < split) for ref in refs)
                    z = xyz[refs[0]] - xyz[i]
                    assert np.linalg.norm(z) > 1e-10
                    if len(refs) == 2:
                        x = xyz[refs[1]] - xyz[i]
                        assert np.linalg.norm(np.cross(z, x)) > 1e-6


def test_browser_harmonics_match_torch_reference():
    import pytest

    torch = pytest.importorskip("torch")
    if not shutil.which("node"):
        pytest.skip("Node.js is required for browser/PyTorch parity")
    spec = importlib.util.spec_from_file_location(
        "mastiff_reference", SCRIPT.parents[1] / "src/apnet_pt/mastiff.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    vectors = np.random.default_rng(13).normal(size=(100, 3))
    code = (
        "const H=require('./scripts/spherical_vis/harmonics.js');"
        f"console.log(JSON.stringify({json.dumps(vectors.tolist())}.map(H.basis)));"
    )
    actual = json.loads(subprocess.check_output(
        ["node", "-e", code], cwd=SCRIPT.parents[1], text=True
    ))
    expected = reference.RacahHarmonics()(torch.tensor(vectors)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
