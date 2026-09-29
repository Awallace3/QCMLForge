import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import qcelemental as qcel
import torch

from apnet_pt import constants
from apnet_pt.mace.schema import QUADRUPOLE_CONVENTION
from apnet_pt.training.smoke import (
    fixture_content_hash,
    load_atomic_smoke_fixture,
    load_pair_smoke_fixture,
)


DATA = Path(__file__).parent / "dataset_data"
PAIR_FIXTURE = DATA / "mace_ap3d3_smoke.pkl"
ATOM_FIXTURE = DATA / "mace_atomic_properties_smoke.pkl"
GENERATOR = Path(__file__).parents[1] / "scripts" / "make_mace_ap3d3_smoke_data.py"


def test_generator_check_mode_and_checked_in_content_is_deterministic():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=GENERATOR.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fixtures are deterministic and current" in result.stdout


def test_pair_fixture_contract_hashes_splits_and_coverage():
    with PAIR_FIXTURE.open("rb") as handle:
        raw = pickle.load(handle)
    dataset = load_pair_smoke_fixture(PAIR_FIXTURE, batch_size=16)

    assert raw["schema"] == "qcmlforge-mace-ap3d3-smoke-v1"
    assert raw["content_hash"] == fixture_content_hash(raw)
    assert raw["units"] == {
        "geometry": "angstrom",
        "labels": "kcal/mol",
        "component_order": ["elst", "exch", "indu", "disp"],
    }
    assert 8 <= len(raw["records"]) <= 16
    assert all(
        record["dimer"].get("format") == "qcel-psi4-text-v1"
        for record in raw["records"]
    )
    assert all(np.asarray(record["labels"]).shape == (4,) for record in raw["records"])
    assert all(
        float(
            qcel.models.Molecule.from_data(record["dimer"]["data"])
            .get_fragment(fragment).molecular_charge
        ) == 0.0
        for record in raw["records"]
        for fragment in (0, 1)
    )
    assert len({record["case"] for record in raw["records"]}) >= 2
    assert {record["range"] for record in raw["records"]} >= {"close", "long"}
    assert [record["id"] for record in raw["records"]] == raw["order"]
    assert set(raw["split_ids"]) == {"train", "test"}
    assert dataset.content_hash == raw["content_hash"]
    assert dataset.split_hash == raw["split_hash"]
    assert dataset.preprocessing_hash == raw["preprocessing_hash"]
    assert dataset.train_batches[0].y.shape[1] == 4


def test_atomic_fixture_complete_targets_conventions_and_flags():
    with ATOM_FIXTURE.open("rb") as handle:
        raw = pickle.load(handle)
    dataset = load_atomic_smoke_fixture(ATOM_FIXTURE)

    assert raw["schema"] == "qcmlforge-mace-atomic-properties-smoke-v1"
    assert raw["content_hash"] == fixture_content_hash(raw)
    assert raw["quadrupole_convention"] == QUADRUPOLE_CONVENTION
    assert set(raw["field_status"]) == {
        "q", "mu", "quadrupole", "hfvr", "valence_width", "alpha", "damping"
    }
    assert {value["classification"] for value in raw["field_status"].values()} == {
        "reference", "wiring_only", "derived_physical"
    }
    assert [record["id"] for record in raw["records"]] == raw["order"]
    for record in raw["records"]:
        assert record["monomer"].get("format") == "qcel-psi4-text-v1"
        monomer = qcel.models.Molecule.from_data(record["monomer"]["data"])
        targets = record["targets"]
        natom = len(monomer.symbols)
        charges = np.asarray(targets["q"])
        assert charges.shape == (natom, 1)
        assert charges.sum() == pytest.approx(
            float(monomer.molecular_charge), abs=1e-6
        )
        assert np.asarray(targets["mu"]).shape == (natom, 3)
        quadrupole = np.asarray(targets["quadrupole"])
        assert quadrupole.shape == (natom, 3, 3)
        assert np.allclose(quadrupole, quadrupole.swapaxes(1, 2), atol=1e-12)
        assert np.allclose(np.trace(quadrupole, axis1=1, axis2=2), 0.0, atol=1e-12)
        for name in ("hfvr", "valence_width", "alpha", "damping"):
            value = np.asarray(targets[name])
            assert value.shape == (natom, 1)
            assert np.isfinite(value).all()
            assert (value > 0).all()
        expected_alpha = (
            constants.polarizability_table[
                torch.as_tensor(monomer.atomic_numbers, dtype=torch.long)
            ]
            .detach().cpu().numpy().reshape(-1, 1)
            * np.abs(np.asarray(targets["hfvr"])) ** (4.0 / 3.0)
        )
        assert np.allclose(targets["alpha"], expected_alpha, atol=1.0e-12)
    assert dataset.content_hash == raw["content_hash"]
    assert dataset.split_hash == raw["split_hash"]
