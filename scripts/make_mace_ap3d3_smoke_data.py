#!/usr/bin/env python3
"""Generate deterministic wiring-only MACE/AP3D3 smoke fixtures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import qcelemental as qcel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apnet_pt import constants  # noqa: E402
from apnet_pt.mace.schema import (  # noqa: E402
    PhysicsConfig,
    QUADRUPOLE_CONVENTION,
)
from apnet_pt.training.smoke import fixture_content_hash  # noqa: E402


DATA = ROOT / "tests" / "dataset_data"
PAIR_OUTPUT = DATA / "mace_ap3d3_smoke.pkl"
ATOM_OUTPUT = DATA / "mace_atomic_properties_smoke.pkl"
HARTREE_TO_KCAL = qcel.constants.conversion_factor("hartree", "kcal/mol")


def _hash_record(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _molecule_record(molecule) -> dict[str, str]:
    """Store only primitive text, never a version-bound QCElemental object."""

    text = molecule.to_string("psi4")
    if "no_com" not in text:
        text += "no_com\n"
    if "no_reorient" not in text:
        text += "no_reorient\n"
    return {"format": "qcel-psi4-text-v1", "data": text}


def _minimum_distance_angstrom(dimer) -> float:
    first = np.asarray(dimer.get_fragment(0).geometry) * constants.au2ang
    second = np.asarray(dimer.get_fragment(1).geometry) * constants.au2ang
    return float(np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2).min())


def _translated_dimer(dimer, shift_angstrom: float):
    fragments = []
    for fragment_index in (0, 1):
        fragment = dimer.get_fragment(fragment_index)
        geometry = np.asarray(fragment.geometry) * constants.au2ang
        if fragment_index == 1:
            geometry = geometry + np.array([shift_angstrom, 0.0, 0.0])
        lines = [
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
            for symbol, (x, y, z) in zip(fragment.symbols, geometry)
        ]
        fragments.append(
            f"{int(fragment.molecular_charge)} "
            f"{int(fragment.molecular_multiplicity)}\n" + "\n".join(lines)
        )
    return qcel.models.Molecule.from_data(
        "\n--\n".join(fragments)
        + "\nunits angstrom\nno_com\nno_reorient\n"
    )


def _water_labels(row) -> np.ndarray:
    return np.array(
        [
            row["SAPT0 ELST ENERGY adz"],
            row["SAPT0 EXCH ENERGY adz"],
            row["SAPT0 IND ENERGY adz"],
            row["SAPT0 DISP ENERGY adz"],
        ],
        dtype=np.float64,
    )


def _benzene_methanol_labels(row) -> np.ndarray:
    return HARTREE_TO_KCAL * np.array(
        [
            row["SAPT0 ELST ENERGY adtz"],
            row["SAPT0 EXCH ENERGY adtz"],
            row["SAPT0 IND ENERGY adtz"],
            row["SAPT0 DISP ENERGY adtz"],
        ],
        dtype=np.float64,
    )


def make_pair_fixture() -> dict:
    water = pd.read_pickle(DATA / "water_dimer_pes.pkl")
    benzene_methanol = pd.read_pickle(DATA / "df_bz_meoh_mbis.pkl")
    records = []
    for index in range(4):
        row = water.iloc[index]
        dimer = row["qcel_molecule"]
        records.append(
            {
                "id": f"water-dimer-{index:02d}",
                "case": "water-dimer-separation-scan",
                "range": "close" if index < 2 else "intermediate",
                "split": "train",
                "dimer": _molecule_record(dimer),
                "labels": _water_labels(row),
                "label_classification": "reference",
                "minimum_distance_angstrom": _minimum_distance_angstrom(dimer),
                "provenance": {
                    "source_fixture": "tests/dataset_data/water_dimer_pes.pkl",
                    "source_row": int(index),
                    "method": "SAPT0/aug-cc-pVDZ",
                },
            }
        )
    for index in range(2):
        row = benzene_methanol.iloc[index]
        dimer = row["qcel_molecule"]
        records.append(
            {
                "id": f"benzene-methanol-{index:02d}",
                "case": "benzene-methanol",
                "range": "intermediate",
                "split": "train" if index == 0 else "test",
                "dimer": _molecule_record(dimer),
                "labels": _benzene_methanol_labels(row),
                "label_classification": "reference",
                "minimum_distance_angstrom": _minimum_distance_angstrom(dimer),
                "provenance": {
                    "source_fixture": "tests/dataset_data/df_bz_meoh_mbis.pkl",
                    "source_row": int(index),
                    "method": "SAPT0/aug-cc-pVTZ",
                },
            }
        )
    source = water.iloc[7]
    for offset, shift in enumerate((10.0, 12.0)):
        dimer = _translated_dimer(source["qcel_molecule"], shift)
        separation = _minimum_distance_angstrom(dimer)
        wiring_labels = _water_labels(source) * (4.0 / separation) ** 6
        records.append(
            {
                "id": f"water-dimer-long-{offset:02d}",
                "case": "water-dimer-long-range-wiring",
                "range": "long",
                "split": "test",
                "dimer": _molecule_record(dimer),
                "labels": wiring_labels,
                "label_classification": "wiring_only",
                "minimum_distance_angstrom": separation,
                "provenance": {
                    "source_fixture": "tests/dataset_data/water_dimer_pes.pkl",
                    "source_row": 7,
                    "transformation": f"translate fragment B by +{shift:.1f} angstrom x",
                    "label_note": "deterministic r^-6 scaling for execution only",
                },
            }
        )
    order = [record["id"] for record in records]
    split_ids = {
        "train": [record["id"] for record in records if record["split"] == "train"],
        "test": [record["id"] for record in records if record["split"] == "test"],
    }
    # Keep records contiguous by split so the split IDs also preserve fixed order.
    records = [
        next(record for record in records if record["id"] == record_id)
        for record_id in split_ids["train"] + split_ids["test"]
    ]
    order = [record["id"] for record in records]
    physics = PhysicsConfig()
    preprocessing = {
        "contract": "primitive-qcel-text-to-ap3-fused-v2",
        "coordinate_unit": "angstrom",
        "labels_retained": "full-Nx4",
        "default_r_cut": 5.0,
        "default_r_cut_im": 8.0,
    }
    fixture = {
        "schema": "qcmlforge-mace-ap3d3-smoke-v1",
        "purpose": "deterministic wiring fixture; not model-quality evidence",
        "units": {
            "geometry": "angstrom",
            "labels": "kcal/mol",
            "component_order": ["elst", "exch", "indu", "disp"],
        },
        "provenance": {
            "generator": "scripts/make_mace_ap3d3_smoke_data.py",
            "sources": [
                "tests/dataset_data/water_dimer_pes.pkl",
                "tests/dataset_data/df_bz_meoh_mbis.pkl",
            ],
            "network_access": False,
        },
        "tolerances": {
            "float64_atol": 1.0e-7,
            "float64_rtol": 1.0e-7,
            "float32_atol": 2.0e-4,
            "float32_rtol": 2.0e-4,
            "checkpoint_reload": "dtype tolerance; exact hash/config",
        },
        "physics_config": asdict(physics),
        "physics_hash": physics.physics_hash,
        "preprocessing": preprocessing,
        "preprocessing_hash": _hash_record(preprocessing),
        "order": order,
        "split_ids": split_ids,
        "split_hash": _hash_record(split_ids),
        "records": records,
    }
    fixture["content_hash"] = fixture_content_hash(fixture)
    return fixture


def _traceless(values: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (values + values.swapaxes(1, 2))
    trace = np.trace(symmetric, axis1=1, axis2=2) / 3.0
    return symmetric - trace[:, None, None] * np.eye(3)[None, :, :]


def _atomic_record(row, fragment, record_id, case, split, source):
    monomer = row["qcel_molecule"].get_fragment(fragment)
    suffix = "A" if fragment == 0 else "B"
    q = np.asarray(row[f"q_{suffix} pbe0/atz"], dtype=np.float64).reshape(-1, 1)
    q += (float(monomer.molecular_charge) - float(q.sum())) / q.shape[0]
    mu = np.asarray(row[f"mu_{suffix} pbe0/atz"], dtype=np.float64)
    quadrupole = _traceless(
        np.asarray(row[f"theta_{suffix} pbe0/atz"], dtype=np.float64)
    )
    hfvr = np.asarray(
        row[f"vol_ratios_{suffix} pbe0/atz"], dtype=np.float64
    ).reshape(-1, 1)
    valence_width = np.asarray(
        row[f"val_widths_{suffix} pbe0/atz"], dtype=np.float64
    ).reshape(-1, 1)
    atomic_numbers = np.asarray(monomer.atomic_numbers, dtype=np.int64)
    free_atom_alpha = constants.polarizability_table.detach().cpu().numpy()[
        atomic_numbers
    ].reshape(-1, 1)
    alpha = free_atom_alpha * np.abs(hfvr) ** (4.0 / 3.0)
    damping = 1.0 / np.maximum(np.abs(valence_width), 1.0e-6)
    return {
        "id": record_id,
        "case": case,
        "split": split,
        "monomer": _molecule_record(monomer),
        "targets": {
            "q": q,
            "mu": mu,
            "quadrupole": quadrupole,
            "hfvr": np.abs(hfvr),
            "valence_width": np.abs(valence_width),
            "alpha": alpha,
            "damping": damping,
        },
        "provenance": {
            "source_fixture": source,
            "source_row": int(row.name),
            "source_fragment": suffix,
            "reference_method": "PBE0/aug-cc-pVTZ MBIS",
        },
    }


def make_atomic_fixture() -> dict:
    water = pd.read_pickle(DATA / "water_dimer_pes.pkl")
    benzene_methanol = pd.read_pickle(DATA / "df_bz_meoh_mbis.pkl")
    records = [
        _atomic_record(
            water.iloc[0], 0, "water-A", "water", "train",
            "tests/dataset_data/water_dimer_pes.pkl",
        ),
        _atomic_record(
            water.iloc[0], 1, "water-B", "water", "train",
            "tests/dataset_data/water_dimer_pes.pkl",
        ),
        _atomic_record(
            benzene_methanol.iloc[0], 0, "benzene", "benzene", "train",
            "tests/dataset_data/df_bz_meoh_mbis.pkl",
        ),
        _atomic_record(
            benzene_methanol.iloc[0], 1, "methanol", "methanol", "test",
            "tests/dataset_data/df_bz_meoh_mbis.pkl",
        ),
    ]
    order = [record["id"] for record in records]
    split_ids = {
        "train": [record["id"] for record in records if record["split"] == "train"],
        "test": [record["id"] for record in records if record["split"] == "test"],
    }
    preprocessing = {
        "contract": "primitive-qcel-text-ap3-atomic-properties-v2",
        "coordinate_unit": "angstrom",
        "quadrupole_projection": "symmetric-traceless",
        "charge_projection": "uniform exact-monomer-total",
    }
    fixture = {
        "schema": "qcmlforge-mace-atomic-properties-smoke-v1",
        "purpose": "approved wiring fixture; not model-quality evidence",
        "units": {
            "geometry": "angstrom",
            "q": "elementary_charge",
            "mu": "elementary_charge*bohr",
            "quadrupole": "elementary_charge*bohr^2",
            "hfvr": "dimensionless",
            "valence_width": "bohr",
            "alpha": "bohr^3",
            "damping": "bohr^-1",
        },
        "quadrupole_convention": QUADRUPOLE_CONVENTION,
        "field_status": {
            "q": {"classification": "reference", "source": "PBE0/atz MBIS"},
            "mu": {"classification": "reference", "source": "PBE0/atz MBIS"},
            "quadrupole": {"classification": "reference", "source": "PBE0/atz MBIS"},
            "hfvr": {"classification": "reference", "source": "PBE0/atz MBIS"},
            "valence_width": {"classification": "reference", "source": "PBE0/atz MBIS"},
            "alpha": {
                "classification": "derived_physical",
                "source": "QCMLForge free-atom table*abs(HFVR)^(4/3)",
            },
            "damping": {
                "classification": "wiring_only",
                "source": "inverse absolute valence width",
            },
        },
        "provenance": {
            "generator": "scripts/make_mace_ap3d3_smoke_data.py",
            "network_access": False,
        },
        "tolerances": {
            "float64_atol": 1.0e-7,
            "float64_rtol": 1.0e-7,
            "float32_atol": 2.0e-4,
            "float32_rtol": 2.0e-4,
        },
        "preprocessing": preprocessing,
        "preprocessing_hash": _hash_record(preprocessing),
        "order": order,
        "split_ids": split_ids,
        "split_hash": _hash_record(split_ids),
        "records": records,
    }
    fixture["content_hash"] = fixture_content_hash(fixture)
    return fixture


def _serialize(value) -> bytes:
    return pickle.dumps(value, protocol=4)


def _write_or_check(path: Path, value, check: bool) -> bool:
    expected = _serialize(value)
    if check:
        if not path.is_file():
            return False
        try:
            with path.open("rb") as handle:
                actual = pickle.load(handle)
        except Exception:
            return False
        return (
            actual.get("content_hash") == value["content_hash"]
            and fixture_content_hash(actual) == value["content_hash"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in fixture bytes differ from deterministic output",
    )
    args = parser.parse_args(argv)
    pair = make_pair_fixture()
    atomic = make_atomic_fixture()
    results = [
        _write_or_check(PAIR_OUTPUT, pair, args.check),
        _write_or_check(ATOM_OUTPUT, atomic, args.check),
    ]
    if not all(results):
        print("smoke fixtures are missing or differ; rerun generator without --check")
        return 1
    if args.check:
        print("fixtures are deterministic and current")
    else:
        print(f"wrote {PAIR_OUTPUT.relative_to(ROOT)} ({pair['content_hash']})")
        print(f"wrote {ATOM_OUTPUT.relative_to(ROOT)} ({atomic['content_hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
