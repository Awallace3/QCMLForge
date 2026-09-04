"""Shared helpers for the TensorFlow <-> PyTorch AP-Net2 parity fixture.

This module is imported by three different interpreters:

* ``make_parity_dimers.py`` in the PyTorch environment (needs torch),
* ``tf_reference_predictions.py`` in the legacy TensorFlow 2.3 / python 3.8
  environment (needs tensorflow), and
* ``tests/test_ap2_tf_parity.py`` in the PyTorch environment again.

It therefore may only depend on ``numpy`` and ``qcelemental``, and must stay
compatible with python 3.8 syntax -- no ``from __future__ import annotations``
tricks are needed, but no match statements or walrus-free-for-alls either.

The parity fixture stores geometries in **Angstrom, float64**.  That is the unit
the processed PyTorch shards hold (as float32) and the unit both featurisers
consume.  Molecules are rebuilt here by dividing by the same ``bohr ->
angstrom`` conversion factor that ``apnet.constants`` and ``apnet_pt`` both take
straight from qcelemental, so the multiplication each featuriser performs on
``Molecule.geometry`` returns to the stored value to within one float64 ulp --
which the float32 cast that follows erases entirely.  ``verify_round_trip``
asserts exactly that rather than assuming it.
"""

import numpy as np
import qcelemental as qcel

AU2ANG = qcel.constants.conversion_factor("bohr", "angstrom")


def _symbols(z_array):
    return [qcel.periodictable.to_E(int(z)) for z in z_array]


def _molecule(symbols, geometry_ang, fragments, fragment_charges):
    """Build a qcel Molecule with validation off so both environments agree.

    ``validate=False`` keeps qcelemental from running molparse, which is what
    would otherwise re-centre, re-orient, or re-order atoms and make the two
    environments disagree for reasons that have nothing to do with the model
    weights.  ``fix_com``/``fix_orientation`` mirror the ``no_com``/
    ``no_reorient`` markers ``apnet.util.dimerdata_to_qcel`` writes.
    """

    return qcel.models.Molecule(
        symbols=symbols,
        geometry=np.asarray(geometry_ang, dtype=np.float64) / AU2ANG,
        fragments=fragments,
        fragment_charges=[float(c) for c in fragment_charges],
        fragment_multiplicities=[1] * len(fragments),
        molecular_charge=float(sum(fragment_charges)),
        molecular_multiplicity=1,
        fix_com=True,
        fix_orientation=True,
        validate=False,
    )


def build_dimer(record):
    n_a = len(record["ZA"])
    n_b = len(record["ZB"])
    return _molecule(
        _symbols(record["ZA"]) + _symbols(record["ZB"]),
        np.concatenate([record["RA"], record["RB"]], axis=0),
        [list(range(n_a)), list(range(n_a, n_a + n_b))],
        [record["TQA"], record["TQB"]],
    )


def build_monomer(z_array, r_ang, total_charge):
    return _molecule(
        _symbols(z_array), r_ang, [list(range(len(z_array)))], [total_charge]
    )


def verify_round_trip(record, molecule):
    """Confirm ``Molecule.geometry * AU2ANG`` recovers the stored float32 values.

    Both featurisers cast to float32 immediately after this multiplication, so
    equality is asserted at float32 precision; the float64 residual is returned
    for the record.
    """

    stored = np.concatenate([record["RA"], record["RB"]], axis=0)
    recovered = np.asarray(molecule.geometry) * AU2ANG
    if not np.array_equal(stored.astype(np.float32), recovered.astype(np.float32)):
        raise AssertionError("Angstrom round trip is not float32-exact")
    return float(np.max(np.abs(stored - recovered)))


def load_dimers(npz_path):
    """Expand the flat parity npz into a list of per-dimer dictionaries."""

    data = np.load(npz_path)
    offsets_a = np.concatenate([[0], np.cumsum(data["sizes_A"])])
    offsets_b = np.concatenate([[0], np.cumsum(data["sizes_B"])])
    records = []
    for i in range(len(data["sizes_A"])):
        sa, ea = offsets_a[i], offsets_a[i + 1]
        sb, eb = offsets_b[i], offsets_b[i + 1]
        records.append(
            {
                "ZA": data["ZA"][sa:ea],
                "ZB": data["ZB"][sb:eb],
                "RA": data["RA"][sa:ea],
                "RB": data["RB"][sb:eb],
                "TQA": int(data["TQA"][i]),
                "TQB": int(data["TQB"][i]),
            }
        )
    labels = data["labels"] if "labels" in data.files else None
    return records, labels
