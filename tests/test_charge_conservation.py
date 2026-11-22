"""Charge conservation must hold to FP32 rounding of the charges, not of their sum.

The atom models predict raw charges, then shift every atom of a molecule by the
same amount so the molecule sums to its formal charge. Raw charges carry an
arbitrary per-atom offset (the paper AP2 atom models put about +1 e on every
atom), so the shift is O(1) and the raw sum is O(natom). Reducing that sum in
FP32 left a net-charge error of ~2e-4 e on a 342-atom receptor, enough to move
AP2 electrostatics by ~0.017 kcal/mol and to make it depend on reduction order
(threads, batch layout, device).
"""

import numpy as np
import pytest
import qcelemental as qcel
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.multipole import enforce_charge_conservation

# Net-charge error allowed per molecule, in e. Independent FP32 rounding of
# ~500 O(1) charges sums to ~1e-6; the old FP32 reduction gave ~1e-4.
NET_CHARGE_ATOL = 2e-5


def _device(use_gpu):
    return torch.device("cuda" if use_gpu else "cpu")


def _offset_raw_charges(natom_per_mol, offset, seed):
    """FP32 raw charges of ~``offset`` e per atom with O(0.5) spread."""
    generator = torch.Generator().manual_seed(seed)
    natom = int(natom_per_mol.sum())
    noise = torch.rand(natom, 1, generator=generator, dtype=torch.float64) - 0.5
    return (offset + noise).to(torch.float32)


def _water_cluster(n_side, total_charge):
    """A ``n_side``**3-water grid as one monomer with the given formal charge."""
    water = np.array(
        [[0.0, 0.0, 0.0], [0.758602, 0.0, 0.504284], [0.260455, 0.0, -0.872893]]
    )
    spacing = 3.1
    grid = np.array(
        [[i, j, k] for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    )
    geometry = (grid[:, None, :] * spacing + water[None, :, :]).reshape(-1, 3)
    return qcel.models.Molecule(
        symbols=["O", "H", "H"] * len(grid),
        geometry=geometry * qcel.constants.conversion_factor("angstrom", "bohr"),
        molecular_charge=total_charge,
        molecular_multiplicity=1,
        fix_com=True,
        fix_orientation=True,
    )


def test_enforce_charge_conservation_sums_to_target(use_gpu):
    device = _device(use_gpu)
    natom_per_mol = torch.tensor([342, 60, 517], device=device)
    total_charge = torch.tensor([-2.0, 2.0, 1.0], device=device)
    molecule_ind = torch.repeat_interleave(
        torch.arange(3, device=device), natom_per_mol
    )
    raw = _offset_raw_charges(natom_per_mol.cpu(), offset=1.0, seed=7).to(device)

    charge = enforce_charge_conservation(
        raw, molecule_ind, natom_per_mol, total_charge
    )

    assert charge.dtype == raw.dtype
    assert charge.shape == raw.shape
    assert charge.device.type == device.type
    net = torch.zeros(3, dtype=torch.float64, device=device).index_add_(
        0, molecule_ind, charge.squeeze(-1).double()
    )
    assert torch.allclose(net, total_charge.double(), rtol=0, atol=NET_CHARGE_ATOL)

    # Every atom in a molecule moves by the same shift, exact up to FP32 rounding.
    shift = (raw.double() - charge.double()).squeeze(-1)
    for mol in range(3):
        mol_shift = shift[molecule_ind == mol]
        assert torch.allclose(
            mol_shift, mol_shift.mean().expand_as(mol_shift), rtol=0, atol=2e-7
        )


def test_enforce_charge_conservation_propagates_gradients():
    natom_per_mol = torch.tensor([4, 3])
    molecule_ind = torch.repeat_interleave(torch.arange(2), natom_per_mol)
    raw = _offset_raw_charges(natom_per_mol, offset=0.3, seed=3).requires_grad_()

    charge = enforce_charge_conservation(
        raw, molecule_ind, natom_per_mol, torch.tensor([0.0, -1.0])
    )
    weights = torch.arange(7, dtype=torch.float32).unsqueeze(1)
    (charge * weights).sum().backward()

    # d(q_i - mean_mol(q))/dq_j projected on the weights: w_j - mean_mol(w).
    expected = weights.clone()
    for mol in range(2):
        mask = molecule_ind == mol
        expected[mask] -= weights[mask].mean()
    assert raw.grad.dtype == torch.float32
    assert torch.allclose(raw.grad, expected, atol=1e-6)


def test_atom_model_conserves_charge_of_large_ion(use_gpu):
    """A 375-atom dianion whose raw charges sit near +1 e per atom."""
    torch.manual_seed(4201)
    atom_model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=use_gpu)
    with torch.no_grad():
        atom_model.model.guess_layer.weight.add_(1.0)
    atom_model.model.eval()

    mol = _water_cluster(n_side=5, total_charge=-2)
    with torch.inference_mode():
        ((charge, _, _, _),) = atom_model.predict_qcel_mols([mol], batch_size=1)

    charge = torch.as_tensor(charge)
    assert charge.shape == (375,)
    assert charge.dtype == torch.float32
    assert abs(charge.double().sum().item() + 2.0) < NET_CHARGE_ATOL
