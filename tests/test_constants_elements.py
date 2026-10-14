import numpy as np
import pandas as pd

from apnet_pt import constants, util


def test_alkali_alkaline_earth_elements_are_allowed():
    for z, sym in {3: "Li", 12: "Mg", 19: "K", 20: "Ca"}.items():
        assert constants.z_to_elem[z] == sym
        assert constants.elem_to_z[sym] == z
        assert constants.elem_to_z[sym.upper()] == z
        assert z in constants.ALLOWED_ELEMENTS
    # appended: the original indices are unchanged
    assert [constants.z_to_ind[z] for z in (1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35)] == list(range(11))
    assert sorted(constants.z_to_ind.values()) == list(range(len(constants.z_to_elem)))


def test_load_dimer_dataset_keeps_metal_ion_rows(tmp_path):
    rows = []
    for zion, q in ((3, 1), (12, 2), (19, 1), (20, 2), (11, 1)):
        rows.append({
            "RA": np.zeros((1, 3)), "RB": np.array([[0.0, 0.0, 2.5], [0.0, 0.96, 2.8], [0.0, -0.96, 2.8]]),
            "ZA": np.array([zion]), "ZB": np.array([8, 1, 1]),
            "TQA": float(q), "TQB": 0.0,
            "Elst_aug": -1.0, "Exch_aug": 1.0, "Ind_aug": -0.5, "Disp_aug": -0.2,
        })
    path = tmp_path / "ions.pkl"
    pd.DataFrame(rows).to_pickle(path)
    out = util.load_dimer_dataset(str(path), columns=["Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug"])
    assert len(out[0]) == 5
