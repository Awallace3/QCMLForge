"""Tests for designed train/test splits.

The trainers' default split is a uniform random permutation seeded at 42. That
is fine for the bulk of the AP3 monomer set and not fine for the rare elements
CLIFF exchange depends on: sodium appears in 38 of 53,168 monomers, and once
charge is a second axis the seed-42 split leaves five (element, charge) cells
with *zero* held-out monomers -- including neutral sodium, which is precisely
the mode whose reference widths are bimodal.

A split manifest makes the split a reviewable artifact. These tests cover the
two things that can go wrong with one: it can be malformed, and it can be stale
(built against a store that has since been rebuilt or reordered). The second is
the dangerous one, because nothing about a stale manifest looks wrong -- it just
trains on a scrambled split.
"""
import inspect
import textwrap

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from apnet_pt import util
from apnet_pt.AtomModels import ap3_atomtype_mpnn


def _monomer(z, r, charge=0):
    return Data(
        x=torch.tensor(z, dtype=torch.long),
        R=torch.tensor(r, dtype=torch.float32),
        total_charge=torch.tensor(float(charge)),
    )


class _FakeMonomerDataset:
    def __init__(self, monomers):
        self._m = list(monomers)

    def __len__(self):
        return len(self._m)

    def len(self):
        return len(self._m)

    def get(self, idx):
        return self._m[idx]

    def __getitem__(self, idx):
        return self._m[idx]


def _dataset(n=8):
    rng = np.random.default_rng(0)
    return _FakeMonomerDataset([
        _monomer([1, 6, 8], rng.normal(size=(3, 3)) * 2.0, charge=i % 3 - 1)
        for i in range(n)
    ])


def _manifest(tmp_path, dataset, splits, name="split.csv",
              fingerprints=None):
    import pandas as pd

    fps = fingerprints or [
        util.datapoint_fingerprint(dataset.get(i)) for i in range(len(splits))
    ]
    path = tmp_path / name
    pd.DataFrame({
        "index": list(range(len(splits))),
        "split": splits,
        "fingerprint": fps,
    }).to_csv(path, index=False)
    return path


# --- fingerprint ----------------------------------------------------------

def test_monomer_fingerprint_is_stable_across_float32_round_trip():
    """The float32 cast must happen before rounding.

    Raw pickles hold float64 coordinates while the processed store holds
    float32, and the gap is ~1e-7 -- enough to straddle a 4-decimal rounding
    boundary. Rounding float64 directly disagreed with the store on ~3.5% of
    monomers, which is indistinguishable from a genuine reordering.
    """
    rng = np.random.default_rng(7)
    z = np.array([1, 6, 7, 8, 17])
    r64 = rng.normal(size=(5, 3)) * 3.0
    r32 = r64.astype(np.float32)
    assert np.abs(r64 - r32.astype(np.float64)).max() > 0
    assert (util.monomer_fingerprint(z, r64, 0)
            == util.monomer_fingerprint(z, r32, 0))


def test_monomer_fingerprint_separates_real_differences():
    z = np.array([1, 6, 8])
    r = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.1, 0.0]])
    base = util.monomer_fingerprint(z, r, 0)
    assert util.monomer_fingerprint(np.array([1, 6, 7]), r, 0) != base
    assert util.monomer_fingerprint(z, r, 1) != base
    moved = r.copy()
    moved[2, 1] += 0.01
    assert util.monomer_fingerprint(z, moved, 0) != base
    # Order matters: two monomers with the same composition but different
    # atom order are different datapoints to the model.
    assert util.monomer_fingerprint(z[::-1], r[::-1], 0) != base


def test_datapoint_fingerprint_matches_monomer_fingerprint():
    ds = _dataset(3)
    d = ds.get(1)
    assert util.datapoint_fingerprint(d) == util.monomer_fingerprint(
        d.x.numpy(), d.R.numpy(), float(d.total_charge)
    )


# --- manifest loading -----------------------------------------------------

def test_load_split_manifest_round_trip(tmp_path):
    ds = _dataset(10)
    path = _manifest(tmp_path, ds, ["train"] * 8 + ["test"] * 2)
    train, test = util.load_split_manifest(path, dataset=ds, print_level=0)
    assert train.tolist() == list(range(8))
    assert test.tolist() == [8, 9]
    assert train.dtype == np.int64 and test.dtype == np.int64


def test_load_split_manifest_detects_a_stale_manifest(tmp_path):
    """The failure this whole mechanism exists to catch."""
    ds = _dataset(10)
    path = _manifest(tmp_path, ds, ["train"] * 8 + ["test"] * 2)
    # Same length, same indices, different data: a rebuilt or reordered store.
    rng = np.random.default_rng(99)
    other = _FakeMonomerDataset([
        _monomer([1, 6, 8], rng.normal(size=(3, 3)) * 2.0) for _ in range(10)
    ])
    with pytest.raises(ValueError, match="does not match this dataset"):
        util.load_split_manifest(path, dataset=other, print_level=0)


def test_load_split_manifest_verify_none_skips_the_check(tmp_path):
    ds = _dataset(10)
    path = _manifest(tmp_path, ds, ["train"] * 8 + ["test"] * 2)
    rng = np.random.default_rng(99)
    other = _FakeMonomerDataset([
        _monomer([1, 6, 8], rng.normal(size=(3, 3))) for _ in range(10)
    ])
    train, test = util.load_split_manifest(
        path, dataset=other, verify="none", print_level=0
    )
    assert len(train) == 8 and len(test) == 2


def test_load_split_manifest_verify_accepts_a_string_count(tmp_path):
    """argparse hands this through as a string, so "4" must mean 4."""
    ds = _dataset(10)
    path = _manifest(tmp_path, ds, ["train"] * 8 + ["test"] * 2)
    train, _ = util.load_split_manifest(
        path, dataset=ds, verify="4", print_level=0
    )
    assert len(train) == 8
    with pytest.raises(ValueError, match="'all', 'none', or an integer"):
        util.load_split_manifest(path, dataset=ds, verify="some",
                                 print_level=0)
    with pytest.raises(ValueError, match="must be positive"):
        util.load_split_manifest(path, dataset=ds, verify=0, print_level=0)


def test_load_split_manifest_rejects_malformed_input(tmp_path):
    import pandas as pd

    ds = _dataset(6)
    good = _manifest(tmp_path, ds, ["train"] * 5 + ["test"])

    missing = tmp_path / "missing.csv"
    pd.read_csv(good).drop(columns=["fingerprint"]).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing column"):
        util.load_split_manifest(missing, dataset=ds, print_level=0)

    bad_value = tmp_path / "bad_value.csv"
    frame = pd.read_csv(good)
    frame.loc[0, "split"] = "validation"
    frame.to_csv(bad_value, index=False)
    with pytest.raises(ValueError, match="unexpected split value"):
        util.load_split_manifest(bad_value, dataset=ds, print_level=0)

    dup = tmp_path / "dup.csv"
    frame = pd.read_csv(good)
    frame.loc[1, "index"] = 0
    frame.to_csv(dup, index=False)
    with pytest.raises(ValueError, match="duplicate index"):
        util.load_split_manifest(dup, dataset=ds, print_level=0)

    one_sided = tmp_path / "one_sided.csv"
    frame = pd.read_csv(good)
    frame["split"] = "train"
    frame.to_csv(one_sided, index=False)
    with pytest.raises(ValueError, match="both must be non-empty"):
        util.load_split_manifest(one_sided, dataset=ds, print_level=0)


def test_load_split_manifest_must_be_exhaustive(tmp_path):
    """A partial manifest silently drops data.

    That is a different experiment from the one the manifest describes, so it
    is an error rather than a quiet subset.
    """
    ds = _dataset(10)
    path = _manifest(tmp_path, ds, ["train"] * 5 + ["test"] * 2)
    with pytest.raises(ValueError, match="must be exhaustive"):
        util.load_split_manifest(path, dataset=ds, print_level=0)


def test_load_split_manifest_rejects_out_of_range_index(tmp_path):
    ds = _dataset(4)
    path = _manifest(tmp_path, ds, ["train"] * 3 + ["test"])
    import pandas as pd
    frame = pd.read_csv(path)
    frame.loc[3, "index"] = 99
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="references index 99"):
        util.load_split_manifest(path, dataset=ds, print_level=0)


def test_load_split_manifest_works_without_a_dataset(tmp_path):
    ds = _dataset(5)
    path = _manifest(tmp_path, ds, ["train"] * 4 + ["test"])
    train, test = util.load_split_manifest(path, print_level=0)
    assert len(train) == 4 and len(test) == 1


# --- trainer wiring -------------------------------------------------------

def test_atom_trainer_declares_explicit_split_indices():
    sig = inspect.signature(ap3_atomtype_mpnn.AtomTypeParamModel.train)
    for name in ("train_indices", "test_indices"):
        assert name in sig.parameters
        assert sig.parameters[name].default is None


def test_atom_trainer_validates_and_records_the_explicit_split():
    src = inspect.getsource(ap3_atomtype_mpnn.AtomTypeParamModel.train)
    src = textwrap.dedent(src)
    # Both sides required: supplying one and letting the other fall back to a
    # random draw would overlap them.
    assert "must be supplied together" in src
    # Leakage between the two sides is the silent failure that matters most.
    assert "leaks" in src
    # And the run record has to say which kind of split was used, or a
    # stratified run is indistinguishable from a uniform one.
    assert '"data/split_kind"' in src


def test_atom_trainer_seeded_uniform_split_is_the_documented_baseline():
    """Pin the default split so the manifest's baseline column stays honest.

    `build_atom_split.py` reproduces this exact draw to report what the uniform
    alternative would have held out. If the trainer's seeding changes, that
    comparison becomes wrong and this test should fail first.
    """
    src = textwrap.dedent(
        inspect.getsource(ap3_atomtype_mpnn.AtomTypeParamModel.train)
    )
    assert "np.random.seed(42)" in src
    assert "random_indices = np.random.permutation(len(self.dataset))" in src


# --- skip_compile plumbing ------------------------------------------------

def test_train_atom_model_captures_skip_compile_before_branches_clobber_it():
    """Regression: the per-model-type branches assign `skip_compile` themselves.

    Every branch does `skip_compile = False`, so reading the parameter after
    them silently discards the caller's request -- the flag would appear to
    work while changing nothing. The request has to be captured first.

    The flag exists because AtomTypeParamModel's forward writes into a slice of
    a mask-filtered tensor (`K_filtered[:, p] += ...`), which Inductor cannot
    guard on; on some torch builds that raises GuardOnDataDependentSymNode
    after the pre-training evaluation.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "train_models.py"
    text = src.read_text()
    tree = ast.parse(text)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_atom_model"
    )
    assert "skip_compile" in {a.arg for a in fn.args.args}

    def line_of(needle):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                seg = ast.get_source_segment(text, node) or ""
                if needle in seg:
                    return node.lineno
        return None

    capture = line_of("skip_compile_requested = skip_compile")
    override = line_of("skip_compile = bool(skip_compile_requested)")
    assert capture is not None, "the caller's skip_compile is never captured"
    assert override is not None, "the captured request is never applied"
    branch_writes = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "skip_compile"
            for t in node.targets
        )
        and node.lineno not in (override,)
    ]
    assert branch_writes, "expected the per-type branches to assign it"
    assert capture < min(branch_writes), (
        f"capture at line {capture} must precede the branch assignments at "
        f"{sorted(branch_writes)}"
    )
    assert override > max(branch_writes), (
        f"override at line {override} must follow the branch assignments at "
        f"{sorted(branch_writes)}"
    )
