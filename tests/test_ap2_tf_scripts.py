"""Focused tests for the AP-Net2 TensorFlow conversion utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import torch
from torch_geometric.data import Data

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts" / "ap2_tf"
sys.path.insert(0, str(SCRIPTS_DIR))
import make_parity_dimers  # noqa: E402


def _dimer(monatomic: bool = False) -> Data:
    n_atoms_a = 1 if monatomic else 2
    return Data(
        ZA=torch.ones(n_atoms_a, dtype=torch.int64),
        ZB=torch.ones(2, dtype=torch.int64),
        RA=torch.zeros((n_atoms_a, 3)),
        RB=torch.zeros((2, 3)),
        total_charge_A=torch.tensor(0),
        total_charge_B=torch.tensor(0),
        y=torch.zeros(4),
    )


def test_collect_keeps_successor_when_last_leading_dimer_is_monatomic(tmp_path):
    shard = tmp_path / "fixture_0.pt"
    torch.save([_dimer(), _dimer(monatomic=True), _dimer()], shard)

    records, provenance, _ = make_parity_dimers.collect(
        tmp_path, "fixture_", samples=2, ensure_monatomic=True
    )

    assert len(records) == 3
    assert [entry["reason"] for entry in provenance] == [
        "leading",
        "leading",
        "follows monatomic",
    ]
