"""Intermolecular cutoff (``r_cut_im``) and smooth switch on APNet3-fused-d3.

The store bakes the short/long-range AB edge split at build time, so a model
cutoff shorter than the store's must be applied in the forward pass.  These
tests pin that a masked 8 A store reproduces a store built natively at the
shorter cutoff, that the optional switch takes each pair to zero at the
cutoff, and that a store cut shorter than the model is refused.
"""

import os

import pytest
import qcelemental as qcel
import torch

import apnet_pt
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_Model

current_file_path = os.path.dirname(os.path.realpath(__file__))
am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"

# Water dimer whose AB pairs span roughly 1.9-4.6 A, so a 3 A cutoff splits them.
water_dimer = qcel.models.Molecule.from_data("""
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
units angstrom
""")


def _model(r_cut_im, sr_switch=False):
    torch.manual_seed(0)
    atom_type = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    elst = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    ap3d3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type.model,
        dimer_prop_model=elst.dimer_model,
        use_precomputed_classical=True,
        ignore_database_null=True,
        use_GPU=False,
        r_cut_im=r_cut_im,
        sr_switch=sr_switch,
    )
    ap3d3.model.eval()
    return ap3d3


def _batch(ap3d3, store_r_cut_im):
    return ap3d3._qcel_example_input(
        [water_dimer], batch_size=1, r_cut=5.0, r_cut_im=store_r_cut_im
    )


def _sr_distances(batch):
    RA = batch.RA.index_select(0, batch.e_ABsr_source)
    RB = batch.RB.index_select(0, batch.e_ABsr_target)
    return torch.linalg.norm(RB - RA, dim=-1)


@pytest.mark.parametrize("sr_switch", [False, True])
def test_masked_store_matches_native_store(sr_switch):
    ap3d3 = _model(3.0, sr_switch=sr_switch)
    wide = _batch(ap3d3, 8.0)
    native = _batch(ap3d3, 3.0)
    dR = _sr_distances(wide)
    assert (dR > 3.0).any() and (dR <= 3.0).any(), "fixture must straddle 3 A"

    with torch.no_grad():
        E_wide = ap3d3.model(wide)[0]
        E_native = ap3d3.model(native)[0]
    assert torch.allclose(E_wide, E_native, atol=1e-6), (E_wide, E_native)


def test_cutoff_changes_the_prediction():
    """The mask is live: 3 A and 8 A differ on the same weights and batch."""
    short = _model(3.0)
    full = _model(8.0)
    full.model.load_state_dict(short.model.state_dict())
    batch = _batch(short, 8.0)
    with torch.no_grad():
        assert not torch.allclose(short.model(batch)[0], full.model(batch)[0])


def test_switch_takes_pair_energy_to_zero_at_cutoff():
    probe = _model(8.0)
    dR = _sr_distances(_batch(probe, 8.0))
    r_far = float(dR.max())

    on = _model(r_far + 1e-4, sr_switch=True)
    off = _model(r_far + 1e-4, sr_switch=False)
    off.model.load_state_dict(on.model.state_dict())
    batch = _batch(on, 8.0)
    with torch.no_grad():
        E_on = on.model(batch)[1]
        E_off = off.model(batch)[1]
    far = int(torch.argmax(_sr_distances(batch)))
    assert E_off[far].abs().max() > 1e-6
    assert E_on[far].abs().max() < 1e-6 * E_off[far].abs().max().clamp_min(1.0)


def test_store_shorter_than_model_is_refused():
    ap3d3 = _model(8.0)
    batch = _batch(ap3d3, 3.0)
    with pytest.raises(ValueError, match="r_cut_im"):
        ap3d3.model(batch)


def test_sr_switch_round_trips_through_config_and_defaults_off():
    assert _model(8.0).model.get_config()["sr_switch"] is False
    assert _model(4.0, sr_switch=True).model.get_config()["sr_switch"] is True


def _reload(ap3d3, path):
    return APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=ap3d3.atom_type_model.model,
        dimer_prop_model=ap3d3.dimer_prop_model,
        pre_trained_model_path=str(path),
        use_precomputed_classical=True,
        ignore_database_null=True,
        use_GPU=False,
    )


def test_load_without_sr_switch_key_defaults_off(tmp_path):
    ap3d3 = _model(8.0)
    path = tmp_path / "old.pt"
    ap3d3.save_model(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"].pop("sr_switch", None)
    torch.save(checkpoint, path)
    assert _reload(ap3d3, path).model.sr_switch is False


def test_reload_keeps_checkpoint_cutoff_and_switch(tmp_path):
    """A 4 A checkpoint loaded without r_cut_im must not revert to 8 A."""
    ap3d3 = _model(4.0, sr_switch=True)
    batch = _batch(ap3d3, 8.0)
    with torch.no_grad():
        E_saved = ap3d3.model(batch)[0]  # materializes the LazyLinear weights
    path = tmp_path / "short.pt"
    ap3d3.save_model(str(path))
    reloaded = _reload(ap3d3, path)
    reloaded.model.eval()
    assert reloaded.model.r_cut_im == 4.0
    assert reloaded.model.sr_switch is True
    with torch.no_grad():
        assert torch.allclose(E_saved, reloaded.model(batch)[0], atol=1e-6)


class _Captured(Exception):
    pass


def test_train_models_forwards_cutoff_and_switch(monkeypatch, tmp_path):
    """--r_cut_im used to be dropped on the APNet3-fused-d3 route."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1]))
    import train_models

    captured = {}

    class _Stub:
        def __init__(self, *args, **kwargs):
            self.model = self
            self.dimer_model = self

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    mods = train_models.AtomPairwiseModels
    monkeypatch.setattr(mods.mtp_mtp, "AtomTypeParamModel", _Stub)
    monkeypatch.setattr(mods.mtp_mtp, "AM_DimerParam_Model", _Stub)
    monkeypatch.setattr(mods.apnet3_d3_fused, "APNet3D3_AtomType_Model", _capture)
    with pytest.raises(_Captured):
        train_models.train_pairwise_model(
            apnet_model_type="APNet3-fused-d3",
            model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path),
            r_cut_im=4.5,
            sr_switch=True,
        )
    assert captured["r_cut_im"] == 4.5
    assert captured["sr_switch"] is True


def test_sr_switch_is_refused_on_other_routes(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1]))
    import train_models

    with pytest.raises(ValueError, match="sr_switch"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path),
            sr_switch=True,
        )


def test_cli_sr_switch_defaults_to_checkpoint():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1]))
    import train_models

    parser = train_models.build_arg_parser()
    assert parser.parse_args([]).sr_switch is None
    assert parser.parse_args(["--sr_switch"]).sr_switch is True
