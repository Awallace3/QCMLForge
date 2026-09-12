import math

import pytest
import torch

from apnet_pt.AtomPairwiseModels.mtp_mtp import cliff_exchange


def _case(dtype=torch.float64):
    return dict(
        RA=torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype),
        RB=torch.tensor([[1.2, -0.4, 0.7]], dtype=dtype),
        e_AB_source=torch.tensor([0]),
        e_AB_target=torch.tensor([0]),
        valence_widths_A=torch.tensor([0.4], dtype=dtype),
        valence_widths_B=torch.tensor([0.5], dtype=dtype),
        K_exch_A=torch.tensor([2.0], dtype=dtype),
        K_exch_B=torch.tensor([1.5], dtype=dtype),
    )


def _anisotropy(dtype=torch.float64):
    return dict(
        dipole_A=torch.tensor([[0.3, -0.2, 0.1]], dtype=dtype),
        dipole_B=torch.tensor([[-0.1, 0.4, 0.2]], dtype=dtype),
        quadrupole_A=torch.tensor(
            [[[0.2, 0.1, 0.0], [0.1, -0.1, 0.03], [0.0, 0.03, -0.1]]],
            dtype=dtype,
        ),
        quadrupole_B=torch.tensor(
            [[[-0.1, 0.02, 0.04], [0.02, 0.3, 0.0], [0.04, 0.0, -0.2]]],
            dtype=dtype,
        ),
        anisotropy_A=torch.tensor([[0.7, -0.5]], dtype=dtype),
        anisotropy_B=torch.tensor([[-0.2, 0.6]], dtype=dtype),
    )


def test_zero_anisotropy_is_exact_isotropic_limit():
    case = _case()
    iso = cliff_exchange(**case)
    angular = _anisotropy()
    angular["anisotropy_A"].zero_()
    angular["anisotropy_B"].zero_()
    got = cliff_exchange(**case, **angular)
    assert torch.equal(got, iso)


def test_anisotropic_exchange_is_rotation_invariant():
    case = _case()
    angular = _anisotropy()
    theta = 0.73
    rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    expected = cliff_exchange(**case, **angular)
    rotated = dict(case)
    rotated["RA"] = case["RA"] @ rotation.T
    rotated["RB"] = case["RB"] @ rotation.T
    transformed = dict(angular)
    transformed["dipole_A"] = angular["dipole_A"] @ rotation.T
    transformed["dipole_B"] = angular["dipole_B"] @ rotation.T
    transformed["quadrupole_A"] = torch.einsum(
        "ij,ajk,lk->ail", rotation, angular["quadrupole_A"], rotation
    )
    transformed["quadrupole_B"] = torch.einsum(
        "ij,ajk,lk->ail", rotation, angular["quadrupole_B"], rotation
    )
    got = cliff_exchange(**rotated, **transformed)
    assert got.item() == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)


def test_anisotropic_exchange_is_monomer_swap_symmetric():
    case = _case()
    angular = _anisotropy()
    expected = cliff_exchange(**case, **angular)
    swapped = dict(
        RA=case["RB"], RB=case["RA"],
        e_AB_source=case["e_AB_target"], e_AB_target=case["e_AB_source"],
        valence_widths_A=case["valence_widths_B"],
        valence_widths_B=case["valence_widths_A"],
        K_exch_A=case["K_exch_B"], K_exch_B=case["K_exch_A"],
        dipole_A=angular["dipole_B"], dipole_B=angular["dipole_A"],
        quadrupole_A=angular["quadrupole_B"],
        quadrupole_B=angular["quadrupole_A"],
        anisotropy_A=angular["anisotropy_B"],
        anisotropy_B=angular["anisotropy_A"],
    )
    got = cliff_exchange(**swapped)
    assert got.item() == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)


def test_partial_anisotropy_inputs_fail_closed():
    case = _case()
    with pytest.raises(ValueError, match="requires dipoles"):
        cliff_exchange(
            **case,
            dipole_A=torch.zeros((1, 3), dtype=torch.float64),
        )
