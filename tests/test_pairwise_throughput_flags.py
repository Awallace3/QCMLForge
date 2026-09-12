"""Tests for the two knobs that decide what a pairwise epoch costs.

Both existed only as hard-coded values before this module. On a V100 the
consequences were measured rather than guessed: one epoch of the CLIFF
classical route over 100k train + 100k validation dimers cost 4,196 s, which
puts a 100-epoch fit at ~117 h and well outside any 8-hour queue.

Two things drove that number and neither was reachable from the command line:

1. ``ap2_fused_module_dataset`` defaults to ``batch_size=16`` and
   ``AM_DimerParam_Model`` never forwarded anything else, so every positive-
   parameter route trained a 1.8M-parameter model 16 dimers at a time. The
   trainer reads ``train_dataset.training_batch_size``, so the knob has to
   reach the *dataset*; setting it on ``train`` alone would be overwritten.
2. ``ds_max_size`` truncates *both* splits, so a 100k-dimer request evaluated
   100k validation dimers every epoch -- as much work as the training pass.

``batch_size`` is not part of the on-disk layout (that is
``datapoint_storage_n_objects``), so changing it does not invalidate a
processed store. A separate validation cap does bound processing, which is why
it is asserted on the raw ``max_size`` and not only on the truncation.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402

import train_models  # noqa: E402


class _FakeFusedDataset:
    """Stands in for ``ap2_fused_module_dataset`` during construction.

    Records the constructor kwargs and supports the slicing the harness does
    after building, so the truncation and the raw cap can be told apart.
    """

    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        self.kwargs = kwargs
        self.n = kwargs.get("max_size") or 1000
        self.training_batch_size = kwargs.get("batch_size")

    def __len__(self):
        return self.n

    def len(self):
        return self.n

    def __repr__(self):
        return f"_FakeFusedDataset({self.n})"

    def get(self, idx):
        """One item, for the element-exclusion scan.

        Only H and C, so nothing is ever excluded and the scan stops when it
        has collected the requested number of survivors.
        """
        import torch
        from torch_geometric.data import Data

        return Data(
            ZA=torch.tensor([1, 6]), ZB=torch.tensor([6, 1])
        )

    def __getitem__(self, idx):
        import copy

        sliced = copy.copy(self)
        if isinstance(idx, slice):
            sliced.n = len(range(*idx.indices(self.n)))
        elif isinstance(idx, (list, tuple)):
            sliced.n = len(idx)
        return sliced


@pytest.fixture
def fake_fused_dataset(monkeypatch):
    _FakeFusedDataset.calls = []
    monkeypatch.setattr(
        mtp_mtp, "ap2_fused_module_dataset", _FakeFusedDataset
    )
    return _FakeFusedDataset


@pytest.fixture
def atom_model():
    """A randomly initialized ``AtomTypeParamNN``, built without checkpoints."""
    return mtp_mtp.AtomTypeParamModel(
        ds_root=None, use_GPU=False, ignore_database_null=True
    ).model


def _build(atom_model, **kwargs):
    return mtp_mtp.CliffClassicalModel(
        atom_model=atom_model,
        ds_root="fake-root",
        use_GPU=False,
        ignore_database_null=False,
        ds_spec_type=2,
        **kwargs,
    )


def _split_calls(fake):
    """The last train/test constructor kwargs.

    The harness calls its ``setup_ds`` twice -- once with the caller's
    ``force_reprocess`` and once with ``False`` -- so the tail of the record is
    the pair that produced ``self.dataset``.
    """
    train = [c for c in fake.calls if c.get("split") == "train"][-1]
    test = [c for c in fake.calls if c.get("split") == "test"][-1]
    return train, test


# ---------------------------------------------------------------------------
# Dataset batch size


def test_dataset_batch_size_is_declared_on_the_model_constructor():
    sig = inspect.signature(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "ds_batch_size" in sig.parameters
    # 16 is the pre-existing effective value: it is what
    # `ap2_fused_module_dataset` defaults to, so declaring it here changes no
    # run that does not ask for something else.
    assert sig.parameters["ds_batch_size"].default == 16
    # The CLIFF routes reach it through **dataset_kwargs, so they must not
    # shadow it.
    for cls in (
        mtp_mtp.CliffExchangeModel,
        mtp_mtp.CliffClassicalModel,
        mtp_mtp.CliffClassicalOverlapModel,
        mtp_mtp.RackersTholeDampingModel,
        mtp_mtp.RackersTholeDampingOverlapModel,
    ):
        params = inspect.signature(cls.__init__).parameters
        assert "ds_batch_size" not in params, cls.__name__


def test_dataset_batch_size_reaches_both_splits(fake_fused_dataset, atom_model):
    _build(atom_model, ds_max_size=100, ds_batch_size=256)
    train, test = _split_calls(fake_fused_dataset)
    assert train["batch_size"] == 256
    assert test["batch_size"] == 256


def test_dataset_batch_size_defaults_to_sixteen(fake_fused_dataset, atom_model):
    model = _build(atom_model, ds_max_size=100)
    train, test = _split_calls(fake_fused_dataset)
    assert train["batch_size"] == 16
    assert test["batch_size"] == 16
    # And it is what `train` will read, which is the value that decides the
    # step count.
    assert model.dataset[0].training_batch_size == 16


def test_train_reads_the_batch_size_off_the_training_dataset():
    """The knob has to be on the dataset, not on ``train``.

    ``train`` unconditionally overwrites any local ``batch_size`` with
    ``train_dataset.training_batch_size``, so a ``train(batch_size=...)``
    parameter would be silently discarded.
    """
    src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.train)
    assert "batch_size = train_dataset.training_batch_size" in src
    assert "batch_size" not in inspect.signature(
        mtp_mtp.AM_DimerParam_Model.train
    ).parameters


@pytest.mark.parametrize("bad", [0, -1, 1.5, "16", None])
def test_dataset_batch_size_rejects_non_positive_integers(
    fake_fused_dataset, atom_model, bad
):
    with pytest.raises((ValueError, TypeError), match="ds_batch_size"):
        _build(atom_model, ds_max_size=100, ds_batch_size=bad)


# ---------------------------------------------------------------------------
# Separate validation cap


def test_validation_cap_is_declared_and_defaults_to_the_shared_cap():
    sig = inspect.signature(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "ds_max_size_val" in sig.parameters
    assert sig.parameters["ds_max_size_val"].default is None


def test_validation_cap_truncates_only_the_test_split(
    fake_fused_dataset, atom_model
):
    model = _build(atom_model, ds_max_size=100, ds_max_size_val=10)
    train_ds, test_ds = model.dataset
    assert len(train_ds) == 100
    assert len(test_ds) == 10


def test_validation_cap_bounds_test_split_processing(
    fake_fused_dataset, atom_model
):
    """The cap must reach ``max_size``, not only the post-build truncation.

    ``max_size`` bounds how much of the raw store gets *processed* on first
    use. A cap applied only by slicing would still process the full 100k
    validation dimers on a host whose store is not built.
    """
    _build(atom_model, ds_max_size=100, ds_max_size_val=10)
    train, test = _split_calls(fake_fused_dataset)
    assert train["max_size"] == 100
    assert test["max_size"] == 10


def test_exclusion_scan_bounds_each_split_separately(
    fake_fused_dataset, atom_model
):
    """The raw cap that bounds processing is derived per split.

    Element exclusion loosens the raw cap so the scan can reach the requested
    number of survivors. Each split gets its own cap from its own request, or
    the validation store would be processed to the training split's bound.
    """
    model = _build(
        atom_model,
        ds_max_size=100,
        ds_max_size_val=10,
        ds_exclude_elements=[11, 17],
        ds_exclude_scan_multiple=3.0,
    )
    train, test = _split_calls(fake_fused_dataset)
    assert train["max_size"] == 300
    assert test["max_size"] == 30
    # And the requested counts, not the loosened raw caps, are what training
    # and validation actually see.
    assert [len(d) for d in model.dataset] == [100, 10]


def test_validation_cap_unset_keeps_the_shared_cap(
    fake_fused_dataset, atom_model
):
    model = _build(atom_model, ds_max_size=100)
    train, test = _split_calls(fake_fused_dataset)
    assert train["max_size"] == 100
    assert test["max_size"] == 100
    assert [len(d) for d in model.dataset] == [100, 100]


def test_validation_cap_rejected_without_a_shared_cap(
    fake_fused_dataset, atom_model
):
    """An unbounded train split with a bounded val split is a full-store build.

    Rejected rather than honored: the combination reads as "small run" and
    costs a 1.6M-dimer processing job.
    """
    with pytest.raises(ValueError, match="ds_max_size_val"):
        _build(atom_model, ds_max_size=None, ds_max_size_val=10)


def test_validation_cap_rejected_on_a_non_split_dataset(
    fake_fused_dataset, atom_model
):
    """spec_type 1 has one store that ``train`` splits by percentage.

    There is no separate validation store to cap, so accepting the flag would
    silently do nothing.
    """
    with pytest.raises(ValueError, match="ds_max_size_val"):
        mtp_mtp.CliffClassicalModel(
            atom_model=atom_model,
            ds_root="fake-root",
            use_GPU=False,
            ignore_database_null=False,
            ds_spec_type=1,
            ds_max_size=100,
            ds_max_size_val=10,
        )


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10"])
def test_validation_cap_rejects_non_positive_integers(
    fake_fused_dataset, atom_model, bad
):
    with pytest.raises((ValueError, TypeError), match="ds_max_size_val"):
        _build(atom_model, ds_max_size=100, ds_max_size_val=bad)


# ---------------------------------------------------------------------------
# train_models.py plumbing


def test_caps_and_batch_size_are_recorded_for_tracking():
    """A capped validation split makes a different experiment.

    The same argument as ``data/excluded_elements``: a run whose validation
    split is a fifth of its training split cannot be compared with one that
    evaluated the full store, and nothing else on the dashboard would say so.
    """
    src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.train)
    assert '"data/train_cap"' in src
    assert '"data/validation_cap"' in src
    assert '"data/batch_size"' in src
    init_src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "self.ds_max_size_val" in init_src


class _FakeHarness:
    """Records constructor and ``train`` kwargs without building anything."""

    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(self)
        self.kwargs = kwargs
        self.train_calls = []
        self.dataset = object()
        self.model = None

    def train(self, **kwargs):
        self.train_calls.append(kwargs)


class _FakeAtomTypeParamWrapper:
    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(self)
        self.kwargs = kwargs
        self.model = None


@pytest.fixture
def cliff_dispatch(monkeypatch):
    _FakeHarness.calls = []
    _FakeAtomTypeParamWrapper.calls = []
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeAtomTypeParamWrapper,
    )
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "CliffClassicalModel",
        _FakeHarness,
    )
    return _FakeHarness


def test_train_pairwise_declares_both_flags():
    sig = inspect.signature(train_models.train_pairwise_model)
    assert sig.parameters["batch_size"].default is None
    assert sig.parameters["ds_max_size_val"].default is None


def test_dispatch_forwards_the_requested_batch_size(tmp_path, cliff_dispatch):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        batch_size=256,
        ds_max_size=100,
    )
    harness = cliff_dispatch.calls[0]
    assert harness.kwargs["ds_batch_size"] == 256
    # It must not arrive as a `train` kwarg: `train` overwrites its own batch
    # size from the dataset, so a value routed there would be dropped.
    assert "batch_size" not in harness.train_calls[0]


def test_dispatch_forwards_the_route_default_batch_size(
    tmp_path, cliff_dispatch
):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        ds_max_size=100,
    )
    assert cliff_dispatch.calls[0].kwargs["ds_batch_size"] == 16


def test_dispatch_forwards_the_validation_cap(tmp_path, cliff_dispatch):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        ds_max_size=100,
        ds_max_size_val=10,
    )
    harness = cliff_dispatch.calls[0]
    assert harness.kwargs["ds_max_size"] == 100
    assert harness.kwargs["ds_max_size_val"] == 10


def test_dispatch_forwards_an_unset_validation_cap_as_none(
    tmp_path, cliff_dispatch
):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        ds_max_size=100,
    )
    assert cliff_dispatch.calls[0].kwargs["ds_max_size_val"] is None


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_dispatch_rejects_a_non_positive_batch_size(tmp_path, cliff_dispatch, bad):
    with pytest.raises((ValueError, TypeError), match="batch_size"):
        train_models.train_pairwise_model(
            apnet_model_type="CliffClassicalModel",
            model_out=str(tmp_path / "out.pt"),
            batch_size=bad,
        )
    # Rejected before any construction, so a misconfigured run costs nothing.
    assert _FakeAtomTypeParamWrapper.calls == []


def test_dispatch_rejects_the_validation_cap_on_other_routes(tmp_path):
    """Only the positive-parameter branch forwards it into the dataset.

    Accepting it on, say, APNet2 would evaluate the full validation split
    while the run record claimed a capped one.
    """
    with pytest.raises(ValueError, match="ds_max_size_val"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "out.pt"),
            ds_max_size=100,
            ds_max_size_val=10,
        )


@pytest.mark.parametrize(
    "flag,value", [("--batch_size", "64"), ("--ds_max_size_val", "10")]
)
def test_atom_routes_reject_the_pairwise_flags(tmp_path, flag, value):
    """The atom routes have their own batch size and a single monomer store.

    Silently ignoring either flag would leave a run record claiming a shape the
    run never had, so the CLI refuses instead.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "train_models.py",
            "--train_am",
            "AtomModel",
            "--am_model_path",
            str(tmp_path / "am.pt"),
            flag,
            value,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert flag in result.stderr
    assert "--train_apnet" in result.stderr


def test_help_advertises_both_flags():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "train_models.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--batch_size" in result.stdout
    assert "--ds_max_size_val" in result.stdout
