"""AtomTypeParamNN hfvr_output_mode: raw (legacy, default) and softplus heads."""

import pytest
import torch
import torch.nn.functional as F

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    AtomTypeParamNN,
    _infer_atomtypeparamnn_from_state_dict,
)


def _head(mode="raw", n_params=2):
    torch.manual_seed(0)
    return AtomTypeParamNN(
        atom_model=AtomMPNN(n_message=1, n_rbf=2, n_neuron=8, n_embed=4, r_cut=5.0),
        n_message=1, n_neuron=8, n_embed=4,
        param_start_mean=[1.6] * n_params, param_start_std=[0.25] * n_params,
        n_params=n_params, freeze_atom_model=True, hfvr_output_mode=mode,
    ).eval()


def test_raw_default_is_the_legacy_output(atomic_batch):
    head = _head()
    assert head.hfvr_output_mode == "raw"
    with torch.no_grad():
        out, raw = head(atomic_batch), head._raw_head_output(atomic_batch)
    assert all(torch.equal(a, b) for a, b in zip(out, raw))


def test_softplus_maps_channel_zero_only(atomic_batch):
    head = _head("softplus")
    with torch.no_grad():
        K = head(atomic_batch)[-1]
        logits = head._raw_head_output(atomic_batch)[-1]
    torch.testing.assert_close(K[:, 0], F.softplus(logits[:, 0]) + head.hfvr_epsilon)
    assert torch.equal(K[:, 1], logits[:, 1])
    assert (K[:, 0] > 0).all()


def test_softplus_init_starts_at_the_physical_guess():
    head = _head("softplus")
    guess = head.guess_layer[0].weight
    torch.testing.assert_close(F.softplus(guess) + head.hfvr_epsilon,
                               _head("raw").guess_layer[0].weight, rtol=1e-5, atol=1e-5)


def test_single_parameter_head(atomic_batch):
    head = _head("softplus", n_params=1)
    with torch.no_grad():
        K = head(atomic_batch)[-1]
        logits = head._raw_head_output(atomic_batch)[-1]
    assert K.shape == logits.shape
    torch.testing.assert_close(K, F.softplus(logits) + head.hfvr_epsilon)


def test_loader_restores_mode_from_config(atomic_batch):
    trained = _head("softplus")
    state, config = trained.state_dict(), trained.get_config()
    assert config["hfvr_output_mode"] == "softplus"
    rebuilt = _infer_atomtypeparamnn_from_state_dict(state, config=config)
    rebuilt.load_state_dict(state)
    rebuilt.eval()
    assert rebuilt.hfvr_output_mode == "softplus"
    with torch.no_grad():
        assert torch.equal(rebuilt(atomic_batch)[-1], trained(atomic_batch)[-1])
    legacy = _infer_atomtypeparamnn_from_state_dict(state)  # config-free stays raw
    assert legacy.hfvr_output_mode == "raw"


def test_constructor_round_trip_from_config(atomic_batch):
    trained = _head("softplus")
    cfg = trained.get_config()
    rebuilt = AtomTypeParamNN(
        atom_model=AtomMPNN(n_message=1, n_rbf=2, n_neuron=8, n_embed=4, r_cut=5.0),
        **{k: v for k, v in cfg.items() if k not in ("atom_model_config",)})
    rebuilt.load_state_dict(trained.state_dict())
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt(atomic_batch)[-1], trained(atomic_batch)[-1])


@pytest.mark.parametrize("bad", [{"hfvr_output_mode": "abs"}, {"hfvr_epsilon": 0.0},
                                 {"hfvr_epsilon": float("nan")}])
def test_invalid_settings_raise(bad):
    kw = dict(hfvr_output_mode="softplus")
    kw.update(bad)
    with pytest.raises(ValueError):
        AtomTypeParamNN(atom_model=AtomMPNN(n_message=1, n_rbf=2, n_neuron=8, n_embed=4),
                        n_message=1, n_neuron=8, n_embed=4, param_start_mean=[1.6],
                        param_start_std=[0.25], n_params=1, **kw)


def test_nonfinite_logits_raise():
    head = _head("softplus")
    with pytest.raises(FloatingPointError):
        head._transform_hfvr(torch.tensor([[float("nan"), 1.0]]))
