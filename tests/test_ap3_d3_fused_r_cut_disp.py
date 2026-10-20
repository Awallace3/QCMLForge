"""A separate, longer cutoff for the APNet3-fused-d3 dispersion readout (``r_cut_disp``).

With r_cut_disp set, elst, exch and ind are the single-cutoff computation on
the pairs within r_cut_im, and only the dispersion readout sees the pairs out
to r_cut_disp (its own distance basis, switch scaled to r_cut_disp, the same
short-range fade).  Off by default, and then nothing changes: no new
parameters, no new config key, the same outputs.
"""

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_Model

from test_ap3_d3_fused_cutoff import (
    _Captured,
    _batch,
    _model,
    _parse_cli,
    _reload,
    _sr_distances,
    at_elst_path,
    at_hf_vw_path,
    am_path,
)


def _split(r_cut_im, r_cut_disp, sr_switch=False, sr_fade=None, no_disp_nn=False):
    import apnet_pt

    torch.manual_seed(0)
    atom_type = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None, use_GPU=False, ignore_database_null=True,
        atom_model_pre_trained_path=am_path, pre_trained_model_path=at_hf_vw_path)
    elst = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None, use_GPU=False, ignore_database_null=True,
        atom_model=atom_type.model, atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path)
    ap3d3 = APNet3D3_AtomType_Model(
        ds_root=None, atom_type_model=atom_type.model, dimer_prop_model=elst.dimer_model,
        use_precomputed_classical=True, ignore_database_null=True, use_GPU=False,
        r_cut_im=r_cut_im, sr_switch=sr_switch, sr_fade=sr_fade, r_cut_disp=r_cut_disp,
        no_disp_nn=no_disp_nn)
    ap3d3.model.eval()
    return ap3d3


def _paired(r_cut_im, r_cut_disp, **kw):
    """A split-cutoff model and a single-cutoff one sharing every common weight."""
    split = _split(r_cut_im, r_cut_disp, **kw)
    single = _model(r_cut_im, **kw)
    batch = _batch(split, 8.0)
    with torch.no_grad():  # materialize the LazyLinear weights before copying
        split.model(batch)
        single.model(batch)
    state = {k: v for k, v in split.model.state_dict().items()
             if not k.startswith("distance_layer_disp.")}
    single.model.load_state_dict(state)
    return split, single, batch


def test_default_adds_no_parameters_or_config_key():
    m = _model(4.0, sr_switch=True).model
    assert m.r_cut_disp is None
    assert "r_cut_disp" not in m.get_config()
    assert not any(k.startswith("distance_layer_disp") for k in m.state_dict())


@pytest.mark.parametrize("sr_switch", [False, True])
def test_elst_exch_ind_equal_the_single_cutoff_model_and_disp_sees_further(sr_switch):
    split, single, batch = _paired(3.0, 8.0, sr_switch=sr_switch)
    dR = _sr_distances(batch)
    assert (dR > 3.0).any() and (dR <= 3.0).any(), "fixture must straddle 3 A"
    with torch.no_grad():
        E_split = split.model(batch)[0]
        E_single = single.model(batch)[0]
    assert torch.equal(E_split[:, :3], E_single[:, :3])
    assert not torch.allclose(E_split[:, 3], E_single[:, 3])


def test_disp_with_r_cut_disp_equal_r_cut_im_matches_the_single_cutoff_model():
    """Same cutoff for both: the dispersion column must reproduce the old one."""
    split, single, batch = _paired(3.0, 3.0, sr_switch=True)
    # the disp basis is a second DistanceLayer; give it the r_cut_im layer's weights
    split.model.distance_layer_disp.load_state_dict(
        split.model.distance_layer_im.state_dict())
    with torch.no_grad():
        E_split = split.model(batch)[0]
        E_single = single.model(batch)[0]
    assert torch.equal(E_split[:, :3], E_single[:, :3])
    assert torch.allclose(E_split[:, 3], E_single[:, 3], atol=1e-7)


def test_switch_and_fade_apply_per_cutoff():
    m = _split(3.0, 8.0, sr_switch=True, sr_fade=(1.0, 1.5)).model
    r = torch.tensor([0.9, 2.0, 2.999999, 3.0, 5.0, 7.999999, 8.0])
    c_im = m._pair_cutoff(r, m.r_cut_im).squeeze(-1)
    c_d = m._pair_cutoff(r, m.r_cut_disp).squeeze(-1)
    assert c_im[0] == 0 and c_d[0] == 0  # the fade acts on both
    assert c_im[3] == 0 and c_d[3] > 0  # r_cut_im switch only on elst/exch/ind
    assert c_d[6] == 0 and c_d[5] < 1e-12  # disp switch reaches zero at r_cut_disp
    assert torch.allclose(c_d[1], torch.tensor(1 / 8.0) * (
        1 - 28 * 0.25**6 + 48 * 0.25**7 - 21 * 0.25**8))


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="r_cut_disp"):
        _split(4.0, 3.0)
    with pytest.raises(ValueError, match="no_disp_nn"):
        _split(4.0, 8.0, no_disp_nn=True)


def test_store_shorter_than_r_cut_disp_is_refused():
    split = _split(3.0, 8.0)
    with pytest.raises(ValueError, match="r_cut_disp"):
        split.model(_batch(split, 3.5))  # fixture pairs end at 3.86 A


def test_round_trip_and_old_checkpoints_have_none(tmp_path):
    split = _split(4.0, 8.0, sr_switch=True, sr_fade=(1.0, 1.5))
    batch = _batch(split, 8.0)
    with torch.no_grad():
        E_saved = split.model(batch)[0]
    path = tmp_path / "split.pt"
    split.save_model(str(path))
    reloaded = _reload(split, path)
    reloaded.model.eval()
    assert reloaded.model.r_cut_disp == 8.0
    assert reloaded.model.get_config()["r_cut_disp"] == 8.0
    with torch.no_grad():
        assert torch.equal(E_saved, reloaded.model(batch)[0])

    plain = _model(4.0, sr_switch=True)
    with torch.no_grad():
        plain.model(batch)
    path = tmp_path / "plain.pt"
    plain.save_model(str(path))
    assert _reload(plain, path).model.r_cut_disp is None


def test_train_models_forwards_r_cut_disp(monkeypatch, tmp_path):
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
            apnet_model_type="APNet3-fused-d3", model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path), r_cut_im=4.0, r_cut_disp=8.0)
    assert captured["r_cut_disp"] == 8.0
    captured.clear()
    with pytest.raises(_Captured):
        train_models.train_pairwise_model(
            apnet_model_type="APNet3-fused-d3", model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path), r_cut_im=4.0)
    assert "r_cut_disp" not in captured
    with pytest.raises(ValueError, match="r_cut_disp"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2", model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path), r_cut_disp=8.0)


def test_cli_r_cut_disp_defaults_to_checkpoint(monkeypatch):
    assert _parse_cli(monkeypatch, []).r_cut_disp is None
    assert _parse_cli(monkeypatch, ["--r_cut_disp", "8.0"]).r_cut_disp == 8.0
