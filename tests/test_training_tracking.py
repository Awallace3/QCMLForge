"""Tests for optional W&B training tracking infrastructure."""

from __future__ import annotations

import importlib
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import qcelemental as qcel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Data

from apnet_pt.distributed_metrics import globally_reduced_mae
from apnet_pt.training_tracking import (
    FileEventTrainingTracker,
    NullTrainingTracker,
    RunContext,
    TrackerBackend,
    WandbConfig,
    WandbTrainingTracker,
    atomic_metric_payload,
    create_training_tracker,
    epoch_metric_payload,
    hirshfeld_metric_payload,
    pairwise_metric_payload,
    parameter_metric_payload,
    run_tracked_distributed,
    run_tracked_single_process,
    scalar_value,
)


def _context(**overrides):
    values = {
        "harness_class": "AtomModel",
        "model_class": "AtomMPNN",
        "model_family": "atomic",
        "output_path": "/private/models/atom.pt",
        "warm_start_path": "/private/pretrained/source.pt",
        "world_size": 2,
        "global_rank": 0,
    }
    values.update(overrides)
    return RunContext(**values)


def _read_events(directory: Path) -> list[dict]:
    event_files = list(directory.glob("tracker-*.jsonl"))
    assert len(event_files) == 1
    return [json.loads(line) for line in event_files[0].read_text().splitlines()]


def _spawn_file_tracker(rank: int, world_size: int, event_directory: str) -> None:
    tracker = create_training_tracker(
        WandbConfig(mode="offline"),
        is_primary=rank == 0,
        run_context=_context(
            global_rank=rank,
            local_rank=rank,
            world_size=world_size,
        ),
        backend=TrackerBackend.FILE_EVENT,
        event_directory=event_directory,
    )
    tracker.start(config={"rank": rank})
    tracker.log({"epoch": 1, "rank": rank})
    tracker.finish()


def _spawn_reducer(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
    profile: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        if profile == "pairwise-4":
            local = ((1, 2, 3, 4, 5), 1) if rank == 0 else ((6, 7, 8, 9, 10), 2)
            result = globally_reduced_mae(local[0], local[1])
        elif profile == "pairwise-3":
            local = ((2, 3, 4, 5), 2) if rank == 0 else ((4, 6, 8, 10), 1)
            result = globally_reduced_mae(local[0], local[1])
        else:
            local = ((2, 12, 54), 2) if rank == 0 else ((1, 6, 27), 1)
            result = globally_reduced_mae(
                local[0], local[1], component_widths=(1, 3, 9)
            )
        if rank == 0:
            Path(result_file).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class _ExternalTrackedHarness:
    def __init__(self):
        self.model = _ToyTrackedModel()
        self.device = torch.device("cpu")
        self.model_save_path = None

    def _create_checkpoint(self, metadata=None):
        return {
            "checkpoint_version": 2,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
            "model_type": "ToyTrackedModel",
            "metadata": metadata or {},
        }

    def train_batches(self):
        return 1.0, 0.25, 0.5, 0.75

    def evaluate_batches(self):
        return 2.0, 0.5, 1.0, 1.5


def _spawn_external_tracker(rank: int, world_size: int, event_directory: str) -> None:
    harness = _ExternalTrackedHarness()

    def train_loop():
        harness.evaluate_batches()
        harness.evaluate_batches()
        harness.train_batches()
        harness.evaluate_batches()

    run_tracked_distributed(
        harness,
        train_loop,
        WandbConfig(mode="offline"),
        rank=rank,
        local_rank=rank,
        model_family="atomic",
        train_dataset=[0, 1],
        validation_dataset=[2, 3],
        effective_batch_size=1,
        world_size=world_size,
        initial_config={"training/epochs": 1},
        backend=TrackerBackend.FILE_EVENT,
        event_directory=event_directory,
    )


def test_config_is_pickle_safe_and_resolves_environment(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "environment-project")
    monkeypatch.setenv("WANDB_ENTITY", "environment-entity")
    monkeypatch.setenv("WANDB_RUN_GROUP", "environment-group")
    monkeypatch.setenv("WANDB_JOB_TYPE", "environment-job")
    config = WandbConfig(mode="offline", tags=("user",))

    restored = pickle.loads(pickle.dumps(config))
    resolved = restored.resolved(("atomic", "user"))

    assert resolved.project == "environment-project"
    assert resolved.entity == "environment-entity"
    assert resolved.group == "environment-group"
    assert resolved.job_type == "environment-job"
    assert resolved.tags == ("user", "atomic")

    explicit = WandbConfig(mode="offline", job_type="explicit-job").resolved()
    assert explicit.job_type == "explicit-job"


def test_train_models_groups_and_names_two_sequential_runs(monkeypatch):
    from train_models import build_wandb_run_configs

    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    args = SimpleNamespace(
        wandb_mode="offline",
        wandb_project="project",
        wandb_entity=None,
        wandb_name="experiment",
        wandb_group=None,
        wandb_tags=("tag",),
        wandb_job_type=None,
        wandb_notes=None,
        wandb_dir=None,
        train_am="AtomModel",
        train_apnet="APNet2",
    )

    atom_config, pairwise_config = build_wandb_run_configs(args)

    assert atom_config.group == pairwise_config.group
    assert atom_config.group.startswith("train-models-")
    assert atom_config.name == "experiment-atom"
    assert pairwise_config.name == "experiment-pairwise"


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid W&B mode"):
        WandbConfig(mode="invalid")
    with pytest.raises(TypeError, match="tuple"):
        WandbConfig(tags=["not", "a", "tuple"])
    with pytest.raises(TypeError, match="project"):
        WandbConfig(project=3)


def test_run_context_sanitizes_paths_and_validates_extra():
    context = _context(extra={"training/epochs": 5})

    config = context.to_config()

    assert config["checkpoint/output_name"] == "atom.pt"
    assert config["checkpoint/warm_start"] == "source.pt"
    assert config["training/epochs"] == 5
    assert "/private" not in json.dumps(config)
    with pytest.raises(TypeError, match="JSON-serializable"):
        _context(extra={"bad": object()})


def test_disabled_and_nonprimary_factory_do_not_import_wandb(monkeypatch):
    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    disabled = create_training_tracker(None, is_primary=True, run_context=_context())
    nonprimary = create_training_tracker(
        WandbConfig(mode="online"),
        is_primary=False,
        run_context=_context(global_rank=1),
    )

    assert isinstance(disabled, NullTrainingTracker)
    assert isinstance(nonprimary, NullTrainingTracker)
    assert not disabled.artifacts_enabled
    disabled.start(config={"ignored": object()})
    disabled.finish()


def test_missing_wandb_has_actionable_error(monkeypatch):
    real_import = importlib.import_module

    def missing_import(name):
        if name == "wandb":
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    tracker = create_training_tracker(
        WandbConfig(mode="online"), is_primary=True, run_context=_context()
    )

    with pytest.raises(ImportError, match=r"qcmlforge\[tracking\]"):
        tracker.start(config={})
    assert not tracker.started
    tracker.finish()


def test_transitive_wandb_import_error_is_preserved(monkeypatch):
    transitive_error = ModuleNotFoundError(
        "No module named 'wandb_dependency'", name="wandb_dependency"
    )

    def missing_dependency(name):
        if name == "wandb":
            raise transitive_error
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    tracker = WandbTrainingTracker(WandbConfig(mode="online"), _context())

    with pytest.raises(ModuleNotFoundError) as exc_info:
        tracker.start(config={})
    assert exc_info.value is transitive_error
    assert exc_info.value.name == "wandb_dependency"


def test_file_event_tracker_lifecycle_and_atomic_aliases(tmp_path):
    tracker = create_training_tracker(
        WandbConfig(mode="offline", project="test-project"),
        is_primary=True,
        run_context=_context(),
        backend=TrackerBackend.FILE_EVENT,
        event_directory=str(tmp_path),
    )
    assert isinstance(tracker, FileEventTrainingTracker)
    assert not tracker.artifacts_enabled

    tracker.start(config={"training/epochs": 2})
    assert tracker.artifacts_enabled
    tracker.update_config({"data/train_samples": 4})
    tracker.define_metrics(["train/loss_sum", "val/loss_sum"])
    tracker.log({"epoch": 1, "train/loss_sum": torch.tensor(2.0)})
    checkpoint = tracker.staging_directory / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = tracker.log_checkpoint(
        checkpoint,
        aliases=("final", "latest", "latest"),
        metadata={"checkpoint_role": "final"},
    )
    tracker.set_summary({"run/status": "completed"})
    tracker.finish(exit_code=0)
    tracker.finish(exit_code=1)

    events = _read_events(tmp_path)
    assert [event["event"] for event in events] == [
        "start",
        "config",
        "define_metrics",
        "log",
        "checkpoint",
        "summary",
        "finish",
    ]
    assert events[4]["aliases"] == ["final", "latest"]
    assert reference.endswith(":latest")
    assert events[-1]["exit_code"] == 0


def test_file_event_tracker_uses_unique_run_and_staging_identity(tmp_path):
    trackers = [
        FileEventTrainingTracker(WandbConfig(mode="offline"), _context(), str(tmp_path))
        for _ in range(2)
    ]

    for tracker in trackers:
        tracker.start(config={})

    events = [
        json.loads(path.read_text().splitlines()[0])
        for path in sorted(tmp_path.glob("tracker-*.jsonl"))
    ]
    assert len(events) == 2
    assert len({event["run_id"] for event in events}) == 2
    assert trackers[0].staging_directory != trackers[1].staging_directory
    assert all(tracker.staging_directory.is_dir() for tracker in trackers)

    for tracker in trackers:
        tracker.finish()


def test_spawned_file_backend_logs_only_global_rank_zero(tmp_path):
    world_size = 2

    mp.spawn(
        _spawn_file_tracker,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    events = _read_events(tmp_path)
    assert [event["event"] for event in events] == ["start", "log", "finish"]
    assert events[0]["config"]["rank"] == 0
    assert events[0]["config"]["runtime/global_rank"] == 0
    assert events[1]["metrics"]["rank"] == 0.0
    assert len(list(tmp_path.glob("staging-*"))) == 1


@pytest.mark.parametrize("profile", ["pairwise-4", "pairwise-3", "atomic"])
def test_world_size_two_reducers_match_global_sample_math(tmp_path, profile):
    result_file = tmp_path / f"{profile}.json"
    mp.spawn(
        _spawn_reducer,
        args=(2, str(tmp_path / f"{profile}.init"), str(result_file), profile),
        nprocs=2,
        join=True,
    )
    distributed = tuple(json.loads(result_file.read_text()))

    if profile == "pairwise-4":
        expected = globally_reduced_mae((7, 9, 11, 13, 15), 3)
    elif profile == "pairwise-3":
        expected = globally_reduced_mae((6, 9, 12, 15), 3)
    else:
        expected = globally_reduced_mae(
            (3, 18, 81), 3, component_widths=(1, 3, 9)
        )
    assert distributed == pytest.approx(expected)


def test_external_ddp_worker_path_has_one_process_safe_tracker_owner(tmp_path):
    mp.spawn(
        _spawn_external_tracker,
        args=(2, str(tmp_path)),
        nprocs=2,
        join=True,
    )

    events = _read_events(tmp_path)
    assert [event["event"] for event in events].count("start") == 1
    assert [event["event"] for event in events].count("finish") == 1
    assert len(list(tmp_path.glob("staging-*"))) == 1
    assert next(event for event in events if event["event"] == "start")["config"][
        "runtime/global_rank"
    ] == 0


def test_file_event_backend_requires_directory():
    with pytest.raises(ValueError, match="event_directory"):
        create_training_tracker(
            WandbConfig(mode="offline"),
            is_primary=True,
            run_context=_context(),
            backend=TrackerBackend.FILE_EVENT,
        )


class _FakeArtifact:
    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, *, name):
        self.files.append((path, name))


class _FakeLoggedArtifact:
    def __init__(self, artifact):
        self.artifact = artifact
        self.wait_calls = 0
        self._committed = False

    def wait(self):
        self.wait_calls += 1
        self._committed = True
        return self

    @property
    def qualified_name(self):
        if not self._committed:
            raise RuntimeError("artifact reference read before commit")
        return f"entity/project/{self.artifact.name}:v3"


class _FakeRun:
    def __init__(self, directory):
        self.id = "run-123"
        self.dir = str(directory)
        self.config = _UpdateRecorder()
        self.summary = _UpdateRecorder()
        self.metrics = []
        self.logged = []
        self.artifacts = []
        self.finished = []

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, metrics):
        self.logged.append(metrics)

    def log_artifact(self, artifact, *, aliases):
        logged = _FakeLoggedArtifact(artifact)
        self.artifacts.append((artifact, aliases, logged))
        return logged

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class _UpdateRecorder(dict):
    def __init__(self):
        super().__init__()
        self.calls = []

    def update(self, values, **kwargs):
        self.calls.append((dict(values), kwargs))
        return super().update(values)


@pytest.mark.parametrize(
    ("mode", "expected_reference", "expected_waits"),
    [
        ("online", "entity/project/qcmlforge-model-run-123:v3", 1),
        ("offline", "qcmlforge-model-run-123:latest", 0),
    ],
)
def test_wandb_tracker_commits_online_and_uses_offline_alias_reference(
    monkeypatch, tmp_path, mode, expected_reference, expected_waits
):
    run = _FakeRun(tmp_path)
    init_calls = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: init_calls.append(kwargs) or run,
        Artifact=_FakeArtifact,
    )
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else real_import(name),
    )
    tracker = WandbTrainingTracker(
        WandbConfig(mode=mode, project="project", tags=("user",)), _context()
    )

    tracker.start(config={"training/epochs": 1})
    tracker.define_metrics(["val/loss_sum"])
    tracker.update_config({"data/train_samples": 2})
    tracker.log({"epoch": 1, "val/loss_sum": 1.25})
    checkpoint = tracker.staging_directory / "model.pt"
    checkpoint.write_bytes(b"model")
    reference = tracker.log_checkpoint(
        checkpoint,
        aliases=("best", "final", "latest"),
        metadata={"source_epoch": 1},
    )
    tracker.set_summary({"run/status": "completed"})
    tracker.finish()

    assert init_calls[0]["resume"] == "never"
    assert init_calls[0]["mode"] == mode
    assert set(init_calls[0]["tags"]) >= {"user", "atomic", "AtomModel"}
    assert run.metrics[-1] == (
        "val/loss_sum",
        {"step_metric": "epoch", "summary": "min"},
    )
    assert run.artifacts[0][1] == ["best", "final", "latest"]
    assert run.artifacts[0][2].wait_calls == expected_waits
    assert reference == expected_reference
    assert run.finished == [0]


class _FatalTrackerError(BaseException):
    pass


def _break_tracker_operation(tracker, operation):
    if operation == "summary":
        def broken_summary(values):
            raise _FatalTrackerError("secondary-summary")

        tracker.set_summary = broken_summary
        return lambda: tracker.set_summary_safely({"run/status": "failed"})

    def broken_finish(*, exit_code=0):
        raise _FatalTrackerError("secondary-finish")

    tracker.finish = broken_finish
    return lambda: tracker.finish_safely(exit_code=1)


@pytest.mark.parametrize("operation", ["summary", "finish"])
def test_safe_wrappers_preserve_active_primary_base_exception(
    tmp_path, capsys, operation
):
    tracker = FileEventTrainingTracker(
        WandbConfig(mode="offline"), _context(), str(tmp_path)
    )
    tracker.start(config={})
    invoke = _break_tracker_operation(tracker, operation)

    with pytest.raises(ValueError, match="primary"):
        try:
            raise ValueError("primary")
        finally:
            invoke()
    captured = capsys.readouterr()
    assert f"secondary-{operation}" in captured.err


@pytest.mark.parametrize("operation", ["summary", "finish"])
def test_safe_wrappers_raise_base_exception_without_primary(tmp_path, operation):
    tracker = FileEventTrainingTracker(
        WandbConfig(mode="offline"), _context(), str(tmp_path)
    )
    tracker.start(config={})
    invoke = _break_tracker_operation(tracker, operation)

    with pytest.raises(_FatalTrackerError, match=f"secondary-{operation}"):
        invoke()


def test_metric_payload_helpers_and_scalar_validation():
    metrics = {}
    metrics.update(epoch_metric_payload(epoch=1, learning_rate=5e-4, is_best=True))
    metrics.update(
        atomic_metric_payload(
            "train",
            loss_sum=torch.tensor(3.0),
            charge_mae=1,
            dipole_mae=2,
            quadrupole_mae=3,
        )
    )
    metrics.update(
        hirshfeld_metric_payload("val", hfvr_mae=0.2, valence_width_mae=0.3)
    )
    metrics.update(
        pairwise_metric_payload(
            "val",
            total_mae=1,
            electrostatics_mae=2,
            exchange_mae=3,
            induction_mae=4,
            dispersion_mae=None,
        )
    )

    assert metrics["train/loss_sum"] == 3.0
    assert metrics["checkpoint/is_best"] is True
    assert metrics["val/mae/valence_width"] == 0.3
    assert metrics["val/mae/total"] == 1.0
    assert "val/mae/dispersion" not in metrics
    assert scalar_value(torch.tensor(2.0), metric_name="x") == 2.0
    with pytest.raises(ValueError, match="scalar"):
        scalar_value(torch.tensor([1.0, 2.0]), metric_name="x")
    with pytest.raises(ValueError, match="finite"):
        scalar_value(float("nan"), metric_name="x")


_PUBLIC_TRAIN_HARNESSES = (
    ("apnet_pt.AtomModels.ap2_atom_model", "AtomModel"),
    ("apnet_pt.AtomModels.ap2_hirshfeld_atom_model", "AtomHirshfeldModel"),
    ("apnet_pt.AtomModels.ap3_atom_model", "AtomInducedDipoleModel"),
    ("apnet_pt.AtomModels.ap3_atom_model_frozen", "InducedDipoleModel"),
    ("apnet_pt.AtomModels.ap3_atomtype_mpnn", "AtomTypeParamModel"),
    ("apnet_pt.AtomPairwiseModels.apnet2", "APNet2Model"),
    ("apnet_pt.AtomPairwiseModels.apnet2_fused", "APNet2_AM_Model"),
    ("apnet_pt.AtomPairwiseModels.apnet3", "APNet3Model"),
    ("apnet_pt.AtomPairwiseModels.apnet3_d3_fused", "APNet3D3_AtomType_Model"),
    ("apnet_pt.AtomPairwiseModels.apnet3_fused", "APNet3_AtomType_Model"),
    (
        "apnet_pt.AtomPairwiseModels.apnet3_fused_variants",
        "APNet3_AtomType_Model",
    ),
    ("apnet_pt.AtomPairwiseModels.dapnet2", "APNet2_dAPNet2Model"),
    ("apnet_pt.AtomPairwiseModels.dapnet2", "dAPNet2Model"),
    ("apnet_pt.AtomPairwiseModels.mtp_mtp", "AM_DimerParam_Model"),
    ("apnet_pt.AtomPairwiseModels.mtp_mtp", "AtomTypeParamModel"),
)


@pytest.mark.parametrize(("module_name", "class_name"), _PUBLIC_TRAIN_HARNESSES)
def test_every_public_train_harness_has_universal_tracking_api(
    module_name, class_name
):
    harness_class = getattr(importlib.import_module(module_name), class_name)
    parameters = inspect.signature(harness_class.train).parameters

    assert parameters["wandb_config"].default is None
    assert parameters["wandb_config"].annotation == WandbConfig | None
    assert parameters["_tracker_backend"].default == TrackerBackend.WANDB
    assert parameters["_tracker_event_directory"].default is None
    assert "run_tracked_single_process" in inspect.getsource(harness_class.train)


class _ToyTrackedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def get_config(self):
        return {"size": 1}


class _ToyTrackedHarness:
    def __init__(self, family):
        self.family = family
        self.model = _ToyTrackedModel()
        self.device = torch.device("cpu")
        self.model_save_path = None
        self.dimer_eval_type = "elst_damping__induced_dipole"

    def _create_checkpoint(self, metadata=None):
        return {
            "checkpoint_version": 2,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
            "model_type": "ToyTrackedModel",
            "metadata": metadata or {},
        }

    def train_batches_single_proc(self):
        with torch.no_grad():
            self.model.weight.add_(1.0)
        return self._result(train=True)

    def evaluate_batches_single_proc(self):
        return self._result(train=False)

    def _result(self, train):
        offset = 0.0 if train else 0.5
        if self.family == "atomic":
            return (2.0 + offset, 0.1 + offset, 0.2 + offset, 0.3 + offset)
        if self.family == "pairwise":
            return (
                2.0 + offset,
                0.1 + offset,
                0.2 + offset,
                0.3 + offset,
                0.4 + offset,
                0.5 + offset,
            )
        return (2.0 + offset, torch.tensor([0.1 + offset, 0.2 + offset]))


def _run_toy_training(harness):
    harness.evaluate_batches_single_proc()
    harness.evaluate_batches_single_proc()
    harness.train_batches_single_proc()
    harness.evaluate_batches_single_proc()


@pytest.mark.parametrize(
    ("family", "expected_metrics"),
    [
        ("atomic", {"charge", "dipole", "quadrupole"}),
        (
            "pairwise",
            {"total", "electrostatics", "exchange", "induction", "dispersion"},
        ),
        ("parameter", {"electrostatics", "induction"}),
    ],
)
def test_file_event_integration_covers_metric_and_checkpoint_families(
    tmp_path, family, expected_metrics
):
    harness = _ToyTrackedHarness(family)
    dataset = list(range(4))

    run_tracked_single_process(
        harness,
        lambda: _run_toy_training(harness),
        WandbConfig(mode="offline"),
        model_family=family,
        train_dataset=dataset,
        validation_dataset=dataset,
        effective_batch_size=2,
        world_size=1,
        initial_config={"training/epochs": 1},
        backend=TrackerBackend.FILE_EVENT,
        event_directory=str(tmp_path),
    )

    events = _read_events(tmp_path)
    logs = [event["metrics"] for event in events if event["event"] == "log"]
    assert [log["epoch"] for log in logs] == [0, 1]
    assert {
        key.removeprefix("train/mae/")
        for key in logs[-1]
        if key.startswith("train/mae/")
    } == expected_metrics
    checkpoints = [event for event in events if event["event"] == "checkpoint"]
    assert checkpoints[0]["aliases"] == ["best"]
    assert checkpoints[-1]["aliases"] == ["final", "latest"]
    assert any(
        event.get("values", {}).get("run/status") == "completed"
        for event in events
        if event["event"] == "summary"
    )


class _TinyIndexableDataset:
    """Small indexable dataset matching the harnesses' numpy-index contract."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray, torch.Tensor)):
            return _TinyIndexableDataset([self.records[int(i)] for i in index])
        return self.records[index]


def _tiny_atomic_record(offset):
    return Data(
        x=torch.tensor([1, 8], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        R=torch.tensor([[0.0, 0.0, 0.0], [0.8 + offset, 0.0, 0.0]]),
        molecule_ind=torch.zeros(2, dtype=torch.long),
        total_charge=torch.tensor(0, dtype=torch.long),
        charges=torch.zeros(2),
        dipoles=torch.zeros((2, 3)),
        quadrupoles=torch.zeros((2, 3, 3)),
    )


def test_real_atom_internal_ddp_logs_one_globally_owned_lifecycle(tmp_path):
    from apnet_pt.AtomModels.ap2_atom_model import AtomModel

    dataset = _TinyIndexableDataset(
        [_tiny_atomic_record(float(index) * 0.01) for index in range(4)]
    )
    event_directory = tmp_path / "events"
    model = AtomModel(
        dataset=dataset,
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )

    model.train(
        n_epochs=1,
        batch_size=1,
        split_percent=0.5,
        model_path=None,
        skip_compile=True,
        shuffle=False,
        dataloader_num_workers=0,
        world_size=2,
        omp_num_threads_per_process=1,
        random_seed=91,
        wandb_config=WandbConfig(mode="offline"),
        _tracker_backend=TrackerBackend.FILE_EVENT,
        _tracker_event_directory=str(event_directory),
    )

    events = _read_events(event_directory)
    assert [event["event"] for event in events].count("start") == 1
    assert [event["event"] for event in events].count("finish") == 1
    logs = [event["metrics"] for event in events if event["event"] == "log"]
    assert [log["epoch"] for log in logs] == [0, 1]
    assert all("val/mae/quadrupole" in log for log in logs)
    assert next(event for event in events if event["event"] == "start")["config"][
        "training/world_size"
    ] == 2


def test_real_atom_harness_training_emits_file_events_and_loadable_artifact(tmp_path):
    """Exercise an actual harness epoch loop, not a synthetic callback loop."""

    from apnet_pt.AtomModels.ap2_atom_model import AtomModel
    from apnet_pt import model_io

    dataset = _TinyIndexableDataset(
        [_tiny_atomic_record(float(index) * 0.01) for index in range(4)]
    )
    event_directory = tmp_path / "events"
    model = AtomModel(
        dataset=dataset,
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )

    model.train(
        n_epochs=1,
        batch_size=2,
        split_percent=0.5,
        model_path=None,
        skip_compile=True,
        shuffle=False,
        dataloader_num_workers=0,
        omp_num_threads_per_process=1,
        wandb_config=WandbConfig(mode="offline"),
        _tracker_backend=TrackerBackend.FILE_EVENT,
        _tracker_event_directory=str(event_directory),
    )

    events = _read_events(event_directory)
    logs = [event["metrics"] for event in events if event["event"] == "log"]
    assert [log["epoch"] for log in logs] == [0, 1]
    assert logs[0]["checkpoint/is_best"] is False
    assert set(logs[-1]) >= {
        "train/mae/charge",
        "val/mae/dipole",
        "val/mae/quadrupole",
        "checkpoint/is_best",
    }
    checkpoints = [event for event in events if event["event"] == "checkpoint"]
    assert checkpoints[0]["aliases"] == ["best", "final", "latest"]
    checkpoint = model_io.load_checkpoint(checkpoints[0]["path"])
    assert model_io.validate_checkpoint(checkpoint, expected_type="AtomMPNN")
    assert events[-1] == {
        "event": "finish",
        "exit_code": 0,
        "pid": events[-1]["pid"],
    }


def test_enabled_tracking_does_not_change_atomic_training_weights(tmp_path):
    from apnet_pt.AtomModels.ap2_atom_model import AtomModel

    dataset = _TinyIndexableDataset(
        [_tiny_atomic_record(float(index) * 0.01) for index in range(4)]
    )
    torch.manual_seed(123)
    disabled_model = AtomModel(
        dataset=dataset,
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )
    torch.manual_seed(123)
    tracked_model = AtomModel(
        dataset=dataset,
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )
    common = dict(
        n_epochs=1,
        batch_size=2,
        split_percent=0.5,
        model_path=None,
        skip_compile=True,
        shuffle=False,
        dataloader_num_workers=0,
        omp_num_threads_per_process=1,
        random_seed=77,
    )

    disabled_model.train(**common)
    tracked_model.train(
        **common,
        wandb_config=WandbConfig(mode="offline"),
        _tracker_backend=TrackerBackend.FILE_EVENT,
        _tracker_event_directory=str(tmp_path),
    )

    assert all(
        torch.equal(disabled, tracked)
        for disabled, tracked in zip(
            disabled_model.model.state_dict().values(),
            tracked_model.model.state_dict().values(),
        )
    )


def test_real_parameter_harness_epoch_zero_artifact_is_loadable(tmp_path):
    from apnet_pt.AtomModels.ap3_atomtype_mpnn import AtomTypeParamModel

    records = []
    for index in range(4):
        record = _tiny_atomic_record(float(index) * 0.01)
        record.volume_ratios = torch.ones(2)
        record.valence_widths = torch.ones(2)
        records.append(record)
    dataset = _TinyIndexableDataset(records)
    event_directory = tmp_path / "events"
    model = AtomTypeParamModel(
        dataset=dataset,
        n_message=1,
        n_neuron=4,
        n_embed=2,
        r_cut=5.0,
        ignore_database_null=True,
        use_GPU=False,
    )

    model.train(
        n_epochs=0,
        batch_size=2,
        split_percent=0.5,
        model_path=None,
        skip_compile=True,
        shuffle=False,
        dataloader_num_workers=0,
        omp_num_threads_per_process=1,
        wandb_config=WandbConfig(mode="offline"),
        _tracker_backend=TrackerBackend.FILE_EVENT,
        _tracker_event_directory=str(event_directory),
    )

    events = _read_events(event_directory)
    logs = [event["metrics"] for event in events if event["event"] == "log"]
    assert [log["epoch"] for log in logs] == [0]
    checkpoint_event = next(
        event for event in events if event["event"] == "checkpoint"
    )
    assert checkpoint_event["aliases"] == ["best", "final", "latest"]
    reloaded = AtomTypeParamModel(
        pre_trained_model_path=checkpoint_event["path"],
        ignore_database_null=True,
        use_GPU=False,
    )
    assert reloaded.model.embed_layer[0].embedding_dim == 2


def test_real_pairwise_harness_training_emits_events_and_embedded_checkpoint(
    tmp_path,
):
    from apnet_pt.AtomModels.ap2_atom_model import AtomModel
    from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
    from apnet_pt.pairwise_datasets import apnet2_module_dataset
    from apnet_pt import model_io

    molecule = qcel.models.Molecule.from_data(
        """
        0 1
        O 0.0 0.0 0.0
        H 0.7 0.0 0.5
        H 0.2 0.0 -0.8
        --
        0 1
        O 3.0 0.0 0.0
        H 3.7 0.0 0.5
        H 3.2 0.0 -0.8
        """
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "raw").mkdir(parents=True)
    atom_model = AtomModel(
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )
    dataset = apnet2_module_dataset(
        root=str(dataset_root),
        spec_type=None,
        force_reprocess=True,
        atom_model=atom_model.model,
        atomic_batch_size=2,
        datapoint_storage_n_objects=4,
        batch_size=2,
        prebatched=False,
        num_devices=1,
        skip_compile=True,
        print_level=0,
        qcel_molecules=[molecule] * 4,
        energy_labels=[[0.0] * 4] * 4,
        in_memory=True,
        random_seed=None,
    )
    pair_model = APNet2Model(
        atom_model=atom_model.model,
        n_message=1,
        n_rbf=2,
        n_neuron=4,
        n_embed=2,
        ignore_database_null=True,
        use_GPU=False,
    )
    event_directory = tmp_path / "events"

    pair_model.train(
        dataset,
        n_epochs=1,
        shuffle=False,
        skip_compile=True,
        dataloader_num_workers=0,
        omp_num_threads_per_process=1,
        wandb_config=WandbConfig(mode="offline"),
        _tracker_backend=TrackerBackend.FILE_EVENT,
        _tracker_event_directory=str(event_directory),
    )

    events = _read_events(event_directory)
    logs = [event["metrics"] for event in events if event["event"] == "log"]
    assert [log["epoch"] for log in logs] == [0, 1]
    assert "val/mae/dispersion" in logs[-1]
    checkpoint_event = next(
        event for event in events if event["event"] == "checkpoint"
    )
    checkpoint = torch.load(checkpoint_event["path"], weights_only=False)
    assert checkpoint["model_type"] == "APNet2_MPNN"
    assert checkpoint["submodels"]["atom_model"]["model_type"] == "AtomMPNN"
    reloaded = APNet2Model(
        pre_trained_model_path=checkpoint_event["path"],
        ignore_database_null=True,
        use_GPU=False,
    )
    reloaded_state = model_io.load_state_dict_from_checkpoint(checkpoint)
    assert all(
        torch.equal(reloaded.model.state_dict()[key], value)
        for key, value in reloaded_state.items()
    )
    assert checkpoint["config"]["n_neuron"] == 4
    assert checkpoint["submodels"]["atom_model"]["config"]["n_neuron"] == 4


def test_no_dispersion_model_omits_dispersion_metric(tmp_path):
    harness = _ToyTrackedHarness("pairwise")
    harness.model.no_disp_nn = True
    dataset = list(range(2))

    run_tracked_single_process(
        harness,
        lambda: _run_toy_training(harness),
        WandbConfig(mode="offline"),
        model_family="pairwise",
        train_dataset=dataset,
        validation_dataset=dataset,
        effective_batch_size=1,
        world_size=1,
        initial_config={"training/epochs": 1},
        backend=TrackerBackend.FILE_EVENT,
        event_directory=str(tmp_path),
    )

    logs = [
        event["metrics"]
        for event in _read_events(tmp_path)
        if event["event"] == "log"
    ]
    assert all("train/mae/dispersion" not in metrics for metrics in logs)
    assert all("val/mae/dispersion" not in metrics for metrics in logs)


def test_harness_failure_records_failed_status_and_cleans_up(tmp_path):
    harness = _ToyTrackedHarness("atomic")

    def fail_training():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        run_tracked_single_process(
            harness,
            fail_training,
            WandbConfig(mode="offline"),
            model_family="atomic",
            train_dataset=[1],
            validation_dataset=[2],
            effective_batch_size=1,
            world_size=1,
            initial_config={"training/epochs": 1},
            backend=TrackerBackend.FILE_EVENT,
            event_directory=str(tmp_path),
        )

    events = _read_events(tmp_path)
    failed_summary = next(
        event["values"]
        for event in events
        if event["event"] == "summary"
        and event["values"].get("run/status") == "failed"
    )
    assert failed_summary["run/error_type"] == "RuntimeError"
    assert events[-1]["event"] == "finish"
    assert events[-1]["exit_code"] == 1
    assert not hasattr(harness, "_training_tracker")
    assert not hasattr(harness, "_training_tracking_state")


def test_parameter_metrics_validate_labels_and_lengths():
    payload = parameter_metric_payload(
        "train", labels=("HFVR", "Valence Width"), mae_values=(0.1, 0.2)
    )
    assert payload == {
        "train/mae/hfvr": 0.1,
        "train/mae/valence_width": 0.2,
    }
    with pytest.raises(ValueError, match="equal lengths"):
        parameter_metric_payload("train", labels=("a",), mae_values=(1, 2))
    with pytest.raises(ValueError, match="Duplicate"):
        parameter_metric_payload(
            "train", labels=("Valence Width", "valence-width"), mae_values=(1, 2)
        )
