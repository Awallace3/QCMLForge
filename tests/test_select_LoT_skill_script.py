import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "mcp.server.fastmcp",
    reason="requires the optional MCP server dependency",
)
pytest.importorskip(
    "psi4",
    reason="requires psi4, which the timing workflow calls directly",
)

from qcml_mcp.ie_time_esimator_script import main


pytestmark = [pytest.mark.mcp, pytest.mark.slow]

# Maybe it's a bit redundant to set up a folder of dimer geoms when there is data in .mols, but the inteded usage is to parse through a database of geometries

current_file_path = os.path.dirname(os.path.realpath(__file__))
test_geoms_path = f"{current_file_path}/test_data_path/test_geoms"

one_geom_path = f"{test_geoms_path}/one_geom"
two_geom_path = f"{test_geoms_path}/two_geom"
many_geom_path = f"{test_geoms_path}/many_geom"

expected_columns = {
    "id", "n_atoms", "qcel_dimer", "qcel_monA", "qcel_monB",
    "dimer_tvars", "monA_tvars", "monB_tvars",
    "Level of Theory", "ERROR ESTIMATES (kcal/mol)",
    "ESTIMATED CPU TIMES (log10(s))",
}


def _check_df_shape_and_cols(df, n_expected_rows, label):
    assert len(df) == n_expected_rows, (
        f"Expected {n_expected_rows} rows for {label}. Got {len(df)}."
    )
    missing = expected_columns - set(df.columns)
    assert not missing, f"Missing columns in {label}: {missing}"
    assert df["n_atoms"].notna().all(), f"NaN in n_atoms for {label}"
    assert df["ERROR ESTIMATES (kcal/mol)"].dtype == np.float64, (
        f"dtype mismatch for {label}"
    )
    assert df["ESTIMATED CPU TIMES (log10(s))"].dtype == np.float64, (
        f"dtype mismatch for {label}"
    )
    # mask = df["Level of Theory"] == "B3LYP-D3/aug-cc-pVTZ/unCP"
    # if mask.any():
    #     assert df.loc[mask, "ERROR ESTIMATES (kcal/mol)"].notna().all(), (
    #         f"dAPNet2 prediction NaN for B3LYP-D3 in {label}"
    #     )


def test_one_geom():
    df = main(
        geom_path=one_geom_path,
        n_threads=4,
        using_cp=True,
        methods=None,
        bases=["aug-cc-pVTZ"],
        auto_download=True,
        )
    _check_df_shape_and_cols(df, 10, "1 geom, 1 basis, 10 methods")


def test_two_geom():
    df = main(
        geom_path=two_geom_path,
        n_threads=4,
        using_cp=True,
        methods=None,
        bases=["aug-cc-pVTZ", "aug-cc-pVQZ"],
        auto_download=True,
        )
    _check_df_shape_and_cols(df, 40, "2 geoms, 2 bases, 10 methods")


def test_many_geom():
    df = main(
        geom_path=many_geom_path,
        n_threads=4,
        using_cp=True,
        methods=None,
        bases=None,
        auto_download=True,
        )
    _check_df_shape_and_cols(df, 420, "7 geoms, 6 bases, 10 methods")


def test_mols_input():
    import qcelemental as qcel

    mol = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
""")
    df = main(
        mols=[mol, mol],
        n_threads=4,
        using_cp=True,
        methods=["MP2", "wB97X-V"],
        bases=["aug-cc-pVTZ"],
        auto_download=True,
        )
    _check_df_shape_and_cols(df, 4, "2 mols, 1 basis, 2 methods")
    assert df["ERROR ESTIMATES (kcal/mol)"].notna().all()
