"""--n_message: pair-model message-passing depth on the APNet3-fused-d3 route."""

import sys
from pathlib import Path

import pytest
import torch

from test_ap3_d3_fused_cutoff import _Captured, _batch, _parse_cli, _reload

import apnet_pt
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_Model
from test_ap3_d3_fused_cutoff import am_path, at_elst_path, at_hf_vw_path

sys.path.insert(0, str(Path(__file__).parents[1]))
import train_models  # noqa: E402


def _model(n_message):
    torch.manual_seed(0)
    atom_type = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None, use_GPU=False, ignore_database_null=True,
        atom_model_pre_trained_path=am_path, pre_trained_model_path=at_hf_vw_path,
    )
    elst = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None, use_GPU=False, ignore_database_null=True,
        atom_model=atom_type.model, atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    ap3d3 = APNet3D3_AtomType_Model(
        ds_root=None, atom_type_model=atom_type.model, dimer_prop_model=elst.dimer_model,
        use_precomputed_classical=True, ignore_database_null=True, use_GPU=False,
        r_cut_im=4.0, sr_switch=True, n_message=n_message,
    )
    ap3d3.model.eval()
    return ap3d3


def _capture_route(monkeypatch, tmp_path, **kwargs):
    captured = {}

    class _Stub:
        def __init__(self, *args, **kw):
            self.model = self
            self.dimer_model = self

    def _capture(*args, **kw):
        captured.update(kw)
        raise _Captured

    mods = train_models.AtomPairwiseModels
    monkeypatch.setattr(mods.mtp_mtp, "AtomTypeParamModel", _Stub)
    monkeypatch.setattr(mods.mtp_mtp, "AM_DimerParam_Model", _Stub)
    monkeypatch.setattr(mods.apnet3_d3_fused, "APNet3D3_AtomType_Model", _capture)
    with pytest.raises(_Captured):
        train_models.train_pairwise_model(
            apnet_model_type="APNet3-fused-d3", model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path), **kwargs,
        )
    return captured


def test_train_models_forwards_n_message(monkeypatch, tmp_path):
    assert _capture_route(monkeypatch, tmp_path, n_message=2)["n_message"] == 2


def test_unset_n_message_keeps_the_model_default(monkeypatch, tmp_path):
    assert "n_message" not in _capture_route(monkeypatch, tmp_path)


def test_n_message_is_refused_on_other_routes(tmp_path):
    with pytest.raises(ValueError, match="n_message"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2", model_out=str(tmp_path / "m.pt"),
            data_dir=str(tmp_path), n_message=2,
        )


def test_cli_n_message_defaults_to_none(monkeypatch):
    assert _parse_cli(monkeypatch, []).n_message is None
    assert _parse_cli(monkeypatch, ["--n_message", "4"]).n_message == 4


@pytest.mark.parametrize("n_message", [1, 4])
def test_n_message_sets_depth_and_round_trips(tmp_path, n_message):
    ap3d3 = _model(n_message)
    assert ap3d3.model.get_config()["n_message"] == n_message
    batch = _batch(ap3d3, 8.0)
    with torch.no_grad():
        E_saved = ap3d3.model(batch)[0]
    path = tmp_path / "m.pt"
    ap3d3.save_model(str(path))
    reloaded = _reload(ap3d3, path)
    reloaded.model.eval()
    assert reloaded.model.n_message == n_message
    with torch.no_grad():
        assert torch.allclose(E_saved, reloaded.model(batch)[0], atol=1e-6)


def test_depth_changes_parameter_count():
    count = {n: sum(p.numel() for p in _model(n).model.parameters()
                    if not isinstance(p, torch.nn.parameter.UninitializedParameter))
             for n in (1, 3)}
    assert count[1] < count[3]


def test_reload_refuses_an_architecture_that_contradicts_the_checkpoint(tmp_path):
    ap3d3 = _model(1)
    path = tmp_path / "m.pt"
    ap3d3.save_model(str(path))
    with pytest.raises(ValueError, match="n_message"):
        APNet3D3_AtomType_Model(
            ds_root=None, atom_type_model=ap3d3.atom_type_model.model,
            dimer_prop_model=ap3d3.dimer_prop_model, pre_trained_model_path=str(path),
            use_precomputed_classical=True, ignore_database_null=True, use_GPU=False,
            n_message=3,
        )
