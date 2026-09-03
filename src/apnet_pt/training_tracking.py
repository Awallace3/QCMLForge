"""Optional experiment tracking support for model training.

This module deliberately does not import :mod:`wandb` at module import time.  The
W&B SDK is an optional training dependency and is loaded only when an enabled
``WandbTrainingTracker`` starts.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

Scalar = int | float | bool
_VALID_MODES = {"disabled", "online", "offline"}


@dataclass(frozen=True)
class WandbConfig:
    """User-facing W&B configuration that is safe to pass through ``mp.spawn``."""

    mode: Literal["disabled", "online", "offline"] = "disabled"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    job_type: str | None = None
    notes: str | None = None
    directory: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            expected_modes = sorted(_VALID_MODES)
            raise ValueError(
                f"Invalid W&B mode {self.mode!r}; expected one of {expected_modes}"
            )
        if not isinstance(self.tags, tuple) or not all(
            isinstance(tag, str) and tag for tag in self.tags
        ):
            raise TypeError("W&B tags must be a tuple of non-empty strings")
        for field_name in (
            "project",
            "entity",
            "name",
            "group",
            "job_type",
            "notes",
            "directory",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"W&B {field_name} must be a string or None")
        if self.job_type == "":
            raise ValueError("W&B job_type must not be empty")

    def resolved(
        self,
        generated_tags: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
    ) -> WandbConfig:
        """Return configuration with standard W&B environment fallbacks applied.

        ``environment`` defaults to ``os.environ`` and exists so callers and
        tests can supply an explicit mapping instead of mutating the process
        environment.
        """

        env = os.environ if environment is None else environment
        tags = tuple(dict.fromkeys((*self.tags, *generated_tags)))
        return replace(
            self,
            project=self.project or env.get("WANDB_PROJECT") or "qcmlforge",
            entity=self.entity or env.get("WANDB_ENTITY"),
            name=self.name or env.get("WANDB_NAME"),
            group=self.group or env.get("WANDB_RUN_GROUP"),
            job_type=self.job_type or env.get("WANDB_JOB_TYPE") or "train",
            notes=self.notes or env.get("WANDB_NOTES"),
            directory=self.directory or env.get("WANDB_DIR"),
            tags=tags,
        )


@dataclass(frozen=True)
class RunContext:
    """Serializable facts describing the training process that owns a run."""

    harness_class: str
    model_class: str
    model_family: Literal["atomic", "pairwise", "parameter"]
    variant: str | None = None
    output_path: str | None = None
    warm_start_path: str | None = None
    world_size: int = 1
    global_rank: int = 0
    local_rank: int | None = None
    device: str | None = None
    dataset_class: str | None = None
    dataset_spec_type: int | None = None
    dataset_type: str | None = None
    train_size: int | None = None
    validation_size: int | None = None
    effective_batch_size: int | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    versions: Mapping[str, str | None] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.harness_class or not self.model_class:
            raise ValueError("RunContext class names must not be empty")
        if self.model_family not in {"atomic", "pairwise", "parameter"}:
            raise ValueError(f"Invalid model family: {self.model_family!r}")
        if self.world_size < 1:
            raise ValueError("world_size must be at least one")
        _ensure_json_serializable(self.to_config())

    @property
    def generated_tags(self) -> tuple[str, ...]:
        tags = [self.model_family, self.harness_class, f"world-size-{self.world_size}"]
        if self.variant:
            tags.append(self.variant)
        return tuple(tags)

    def to_config(self) -> dict[str, Any]:
        """Convert context to the stable, flat W&B configuration namespace."""

        output_name = Path(self.output_path).name if self.output_path else None
        warm_start = Path(self.warm_start_path).name if self.warm_start_path else None
        config: dict[str, Any] = {
            "model/harness_class": self.harness_class,
            "model/module_class": self.model_class,
            "model/family": self.model_family,
            "training/variant": self.variant,
            "training/world_size": self.world_size,
            "runtime/global_rank": self.global_rank,
            "runtime/local_rank": self.local_rank,
            "runtime/device": self.device,
            "data/dataset_class": self.dataset_class,
            "data/spec_type": self.dataset_spec_type,
            "data/dataset_type": self.dataset_type,
            "data/train_samples": self.train_size,
            "data/val_samples": self.validation_size,
            "training/effective_batch_size": self.effective_batch_size,
            "checkpoint/output_name": output_name,
            "checkpoint/warm_start": warm_start,
            "runtime/git_commit": self.git_commit,
            "runtime/git_dirty": self.git_dirty,
        }
        config.update({f"runtime/{key}": value for key, value in self.versions.items()})
        config.update(self.extra)
        return {key: value for key, value in config.items() if value is not None}


class TrackerBackend(str, Enum):
    """Pickle-safe backend selector used by production and spawned tests."""

    WANDB = "wandb"
    FILE_EVENT = "file-event"


class TrainingTracker(Protocol):
    """Minimal tracker contract consumed by training harnesses."""

    @property
    def staging_directory(self) -> Path: ...

    @property
    def started(self) -> bool: ...

    @property
    def artifacts_enabled(self) -> bool: ...

    def start(self, *, config: Mapping[str, Any]) -> None: ...

    def update_config(self, values: Mapping[str, Any]) -> None: ...

    def define_metrics(self, metric_names: Sequence[str]) -> None: ...

    def log(self, metrics: Mapping[str, Scalar]) -> None: ...

    def log_checkpoint(
        self,
        path: Path,
        *,
        aliases: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str: ...

    def set_summary(self, values: Mapping[str, Scalar | str | bool]) -> None: ...

    def set_summary_safely(
        self, values: Mapping[str, Scalar | str | bool]
    ) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...

    def finish_safely(self, *, exit_code: int = 0) -> None: ...


class _TrackerState(str, Enum):
    NEW = "new"
    STARTED = "started"
    FINISHED = "finished"


class _BaseTrainingTracker:
    """State management and exception-safe operations shared by real trackers."""

    def __init__(self, wandb_config: WandbConfig, run_context: RunContext) -> None:
        resolved_config = wandb_config.resolved(run_context.generated_tags)
        if resolved_config.name is None:
            output_stem = (
                Path(run_context.output_path).stem
                if run_context.output_path
                else "train"
            )
            resolved_config = replace(
                resolved_config,
                name=f"{run_context.harness_class}-{output_stem}",
            )
        self.wandb_config = resolved_config
        self.run_context = run_context
        self._state = _TrackerState.NEW
        self._staging_directory: Path | None = None

    @property
    def started(self) -> bool:
        return self._state == _TrackerState.STARTED

    @property
    def artifacts_enabled(self) -> bool:
        return self.started

    @property
    def staging_directory(self) -> Path:
        if not self.started or self._staging_directory is None:
            raise RuntimeError("Tracker staging directory is unavailable before start")
        return self._staging_directory

    def _require_started(self) -> None:
        if not self.started:
            raise RuntimeError("Training tracker has not been started")

    def set_summary_safely(
        self, values: Mapping[str, Scalar | str | bool]
    ) -> None:
        active_exception = sys.exc_info()[0] is not None
        try:
            self.set_summary(values)
        except BaseException as exc:
            if active_exception:
                print(
                    f"W&B summary update failed while handling an error: {exc}",
                    file=sys.stderr,
                )
                return
            raise

    def finish_safely(self, *, exit_code: int = 0) -> None:
        active_exception = sys.exc_info()[0] is not None
        try:
            self.finish(exit_code=exit_code)
        except BaseException as exc:
            if active_exception:
                print(
                    f"W&B finish failed while handling an error: {exc}",
                    file=sys.stderr,
                )
                return
            raise


class NullTrainingTracker:
    """Dependency-free no-op tracker used when tracking is disabled/nonprimary."""

    @property
    def staging_directory(self) -> Path:
        raise RuntimeError("Artifacts are disabled for the null tracker")

    @property
    def started(self) -> bool:
        return False

    @property
    def artifacts_enabled(self) -> bool:
        return False

    def start(self, *, config: Mapping[str, Any]) -> None:
        return None

    def update_config(self, values: Mapping[str, Any]) -> None:
        return None

    def define_metrics(self, metric_names: Sequence[str]) -> None:
        return None

    def log(self, metrics: Mapping[str, Scalar]) -> None:
        return None

    def log_checkpoint(
        self,
        path: Path,
        *,
        aliases: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str:
        return ""

    def set_summary(self, values: Mapping[str, Scalar | str | bool]) -> None:
        return None

    def set_summary_safely(
        self, values: Mapping[str, Scalar | str | bool]
    ) -> None:
        return None

    def finish(self, *, exit_code: int = 0) -> None:
        return None

    def finish_safely(self, *, exit_code: int = 0) -> None:
        return None


class WandbTrainingTracker(_BaseTrainingTracker):
    """W&B-backed tracker with a lazy SDK import."""

    def __init__(
        self,
        wandb_config: WandbConfig,
        run_context: RunContext,
        module_loader: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        super().__init__(wandb_config, run_context)
        self._module_loader = module_loader
        self._wandb: Any = None
        self._run: Any = None

    def start(self, *, config: Mapping[str, Any]) -> None:
        if self.started:
            raise RuntimeError("Training tracker is already started")
        if self._state == _TrackerState.FINISHED:
            raise RuntimeError("Finished training trackers cannot be restarted")
        _ensure_json_serializable(config)
        initial_config = {**self.run_context.to_config(), **dict(config)}
        _ensure_json_serializable(initial_config)
        try:
            self._wandb = self._module_loader("wandb")
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            raise ImportError(
                "W&B tracking requires the optional dependency; install it with "
                "`pip install 'qcmlforge[tracking]'`."
            ) from exc

        cfg = self.wandb_config
        self._run = self._wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.name,
            group=cfg.group,
            tags=list(cfg.tags),
            job_type=cfg.job_type,
            notes=cfg.notes,
            dir=cfg.directory,
            mode=cfg.mode,
            config=initial_config,
            id=cfg.run_id,
            resume="never",
        )
        if self._run is None:
            raise RuntimeError("wandb.init() did not return a run")
        self._state = _TrackerState.STARTED
        self._staging_directory = Path(self._run.dir) / "qcmlforge-checkpoints"
        self._staging_directory.mkdir(parents=True, exist_ok=True)

    def update_config(self, values: Mapping[str, Any]) -> None:
        self._require_started()
        _ensure_json_serializable(values)
        self._run.config.update(dict(values), allow_val_change=True)

    def define_metrics(self, metric_names: Sequence[str]) -> None:
        self._require_started()
        self._run.define_metric("epoch")
        for name in metric_names:
            kwargs: dict[str, Any] = {"step_metric": "epoch"}
            if name in {"val/loss_sum", "val/loss_mean", "val/mae/total"}:
                kwargs["summary"] = "min"
            self._run.define_metric(name, **kwargs)

    def log(self, metrics: Mapping[str, Scalar]) -> None:
        self._require_started()
        self._run.log(_validated_metrics(metrics))

    def log_checkpoint(
        self,
        path: Path,
        *,
        aliases: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str:
        self._require_started()
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        clean_aliases = _validated_aliases(aliases)
        _ensure_json_serializable(metadata)
        artifact_name = f"qcmlforge-model-{self._run.id}"
        artifact = self._wandb.Artifact(
            artifact_name, type="model", metadata=dict(metadata)
        )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
        logged = self._run.log_artifact(artifact, aliases=list(clean_aliases))
        if self.wandb_config.mode != "online":
            return f"{artifact_name}:{clean_aliases[-1]}"
        committed = logged.wait()
        committed = committed or logged
        return str(
            getattr(committed, "qualified_name", None)
            or getattr(committed, "name", None)
            or f"{artifact_name}:{clean_aliases[-1]}"
        )

    def set_summary(self, values: Mapping[str, Scalar | str | bool]) -> None:
        self._require_started()
        _ensure_json_serializable(values)
        self._run.summary.update(dict(values))

    def finish(self, *, exit_code: int = 0) -> None:
        if self._state != _TrackerState.STARTED:
            return
        try:
            self._run.finish(exit_code=exit_code)
        finally:
            self._state = _TrackerState.FINISHED


class FileEventTrainingTracker(_BaseTrainingTracker):
    """Process-safe JSONL tracker used by unit and spawned integration tests."""

    def __init__(
        self,
        wandb_config: WandbConfig,
        run_context: RunContext,
        event_directory: str,
    ) -> None:
        super().__init__(wandb_config, run_context)
        if not event_directory:
            raise ValueError("File-event tracking requires event_directory")
        self._event_directory = Path(event_directory)
        self._event_file: Path | None = None
        self._run_id: str | None = None

    @property
    def event_file(self) -> Path:
        if self._event_file is None:
            raise RuntimeError("File-event tracker has not been started")
        return self._event_file

    def _write_event(self, event: str, **payload: Any) -> None:
        record = {"event": event, "pid": os.getpid(), **payload}
        _ensure_json_serializable(record)
        with self.event_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def start(self, *, config: Mapping[str, Any]) -> None:
        if self.started:
            raise RuntimeError("Training tracker is already started")
        if self._state == _TrackerState.FINISHED:
            raise RuntimeError("Finished training trackers cannot be restarted")
        initial_config = {**self.run_context.to_config(), **dict(config)}
        _ensure_json_serializable(initial_config)
        self._event_directory.mkdir(parents=True, exist_ok=True)
        identity = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._run_id = f"file-{identity}"
        self._event_file = self._event_directory / f"tracker-{identity}.jsonl"
        self._event_file.touch(exist_ok=False)
        self._staging_directory = self._event_directory / f"staging-{identity}"
        self._staging_directory.mkdir(parents=True, exist_ok=False)
        self._state = _TrackerState.STARTED
        self._write_event(
            "start",
            config=initial_config,
            wandb_config=asdict(self.wandb_config),
            run_id=self._run_id,
        )

    def update_config(self, values: Mapping[str, Any]) -> None:
        self._require_started()
        _ensure_json_serializable(values)
        self._write_event("config", values=dict(values))

    def define_metrics(self, metric_names: Sequence[str]) -> None:
        self._require_started()
        if not all(isinstance(name, str) and name for name in metric_names):
            raise ValueError("Metric names must be non-empty strings")
        self._write_event("define_metrics", names=list(metric_names))

    def log(self, metrics: Mapping[str, Scalar]) -> None:
        self._require_started()
        self._write_event("log", metrics=_validated_metrics(metrics))

    def log_checkpoint(
        self,
        path: Path,
        *,
        aliases: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str:
        self._require_started()
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        clean_aliases = _validated_aliases(aliases)
        _ensure_json_serializable(metadata)
        reference = f"qcmlforge-model-{self._run_id}:{clean_aliases[-1]}"
        self._write_event(
            "checkpoint",
            path=str(checkpoint_path),
            aliases=list(clean_aliases),
            metadata=dict(metadata),
            reference=reference,
        )
        return reference

    def set_summary(self, values: Mapping[str, Scalar | str | bool]) -> None:
        self._require_started()
        _ensure_json_serializable(values)
        self._write_event("summary", values=dict(values))

    def finish(self, *, exit_code: int = 0) -> None:
        if self._state != _TrackerState.STARTED:
            return
        try:
            self._write_event("finish", exit_code=int(exit_code))
        finally:
            self._state = _TrackerState.FINISHED


def create_training_tracker(
    wandb_config: WandbConfig | None,
    *,
    is_primary: bool,
    run_context: RunContext,
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> TrainingTracker:
    """Create a rank-safe tracker without importing the optional W&B SDK."""

    config = wandb_config or WandbConfig()
    if not is_primary or config.mode == "disabled":
        return NullTrainingTracker()
    try:
        selected_backend = TrackerBackend(backend)
    except ValueError as exc:
        raise ValueError(f"Unsupported tracker backend: {backend!r}") from exc
    if selected_backend == TrackerBackend.FILE_EVENT:
        if event_directory is None:
            raise ValueError("File-event backend requires event_directory")
        return FileEventTrainingTracker(config, run_context, event_directory)
    return WandbTrainingTracker(config, run_context, module_loader)


@contextmanager
def harness_tracking(
    harness: Any,
    wandb_config: WandbConfig | None,
    *,
    model_family: Literal["atomic", "pairwise", "parameter"],
    train_dataset: Any,
    validation_dataset: Any,
    effective_batch_size: int,
    world_size: int,
    initial_config: Mapping[str, Any],
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
    is_primary: bool = True,
    global_rank: int = 0,
    local_rank: int | None = 0,
    variant: str = "single-process",
) -> Generator[TrainingTracker, None, None]:
    """Own a complete rank-aware tracker lifecycle for a model harness."""

    model = getattr(harness, "model", None)
    context = RunContext(
        harness_class=harness.__class__.__name__,
        model_class=model.__class__.__name__ if model is not None else "UnknownModel",
        model_family=model_family,
        variant=variant,
        output_path=getattr(harness, "model_save_path", None),
        warm_start_path=getattr(harness, "pre_trained_model_path", None),
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        device=str(getattr(harness, "device", "cpu")),
        dataset_class=train_dataset.__class__.__name__,
        dataset_spec_type=getattr(harness, "ds_spec_type", None),
        dataset_type=getattr(harness, "ds_type", None),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        effective_batch_size=effective_batch_size,
    )
    tracker = create_training_tracker(
        wandb_config,
        is_primary=is_primary,
        run_context=context,
        backend=backend,
        event_directory=event_directory,
    )
    completed = False
    started_at = time.perf_counter()
    previous_tracker = getattr(harness, "_training_tracker", None)
    try:
        tracker.start(config=_resolved_training_config(harness, initial_config))
        harness._training_tracker = tracker
        if not tracker.started:
            yield tracker
            completed = True
            return
        harness._training_tracking_state = {
            "best_path": stage_harness_checkpoint(
                tracker,
                harness,
                role="best",
                epoch=0,
                validation_loss=None,
            ),
            "best_epoch": 0,
            "best_validation_loss": None,
            "final_path": None,
            "last_validation_loss": None,
            "epochs_completed": 0,
            "defined_metrics": None,
            "config_refreshed": False,
        }
        yield tracker
        state = harness._training_tracking_state
        # The staged best checkpoint already holds the final weights whenever the
        # last completed epoch was also the best one, so only serialize a second
        # checkpoint when training ended on weights worse than the best.
        if state["best_epoch"] == state["epochs_completed"]:
            final_path = state["best_path"]
        elif state["final_path"] is not None:
            # Staged by stage_final_weights() before the loop restored its best
            # weights, so it holds the real final-epoch weights.
            final_path = state["final_path"]
        else:
            final_path = stage_harness_checkpoint(
                tracker,
                harness,
                role="final",
                epoch=state["epochs_completed"],
                validation_loss=state["last_validation_loss"],
            )
        publish_harness_artifacts(
            tracker,
            best_path=state["best_path"],
            final_path=final_path,
            best_epoch=state["best_epoch"],
            best_validation_loss=state["best_validation_loss"],
        )
        tracker.set_summary(
            {
                "run/status": "completed",
                "run/epochs_completed": state["epochs_completed"],
                "runtime/total_seconds": time.perf_counter() - started_at,
            }
        )
        completed = True
    except BaseException as exc:
        if tracker.started:
            tracker.set_summary_safely(
                {
                    "run/status": "failed",
                    "run/error_type": exc.__class__.__name__,
                    "run/error_message": str(exc)[:500],
                }
            )
        raise
    finally:
        try:
            del harness._training_tracking_state
        except AttributeError:
            pass
        if previous_tracker is None:
            try:
                del harness._training_tracker
            except AttributeError:
                pass
        else:
            harness._training_tracker = previous_tracker
        tracker.finish_safely(exit_code=0 if completed else 1)


def run_tracked_single_process(
    harness: Any,
    training_callable: Callable[[], Any],
    wandb_config: WandbConfig | None,
    *,
    model_family: Literal["atomic", "pairwise", "parameter"],
    train_dataset: Any,
    validation_dataset: Any,
    effective_batch_size: int,
    world_size: int,
    initial_config: Mapping[str, Any],
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
) -> Any:
    """Execute a single-process training callable inside a tracker lifecycle."""

    with harness_tracking(
        harness,
        wandb_config,
        model_family=model_family,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        effective_batch_size=effective_batch_size,
        world_size=world_size,
        initial_config=initial_config,
        backend=backend,
        event_directory=event_directory,
    ):
        return training_callable()


def configure_distributed_tracking(
    harness: Any,
    wandb_config: WandbConfig | None,
    *,
    model_family: Literal["atomic", "pairwise", "parameter"],
    initial_config: Mapping[str, Any],
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
    local_rank: int | None = None,
) -> None:
    """Attach only pickle-safe tracking descriptors before a DDP launch."""

    descriptor = {
        "wandb_config": wandb_config,
        "model_family": model_family,
        "initial_config": dict(initial_config),
        "backend": TrackerBackend(backend),
        "event_directory": event_directory,
        "local_rank": local_rank,
    }
    # Fail in the parent rather than after spawning when callers pass unsafe config.
    pickle_values = {
        key: value
        for key, value in descriptor.items()
        if key not in {"wandb_config", "backend"}
    }
    _ensure_json_serializable(pickle_values)
    harness._distributed_tracking_descriptor = descriptor


def tracked_ddp_worker(rank: int, ddp_callable: Callable[..., Any], *args: Any) -> Any:
    """Run one internal-spawn DDP rank with global-rank-zero tracking ownership."""

    harness = ddp_callable.__self__
    descriptor = getattr(harness, "_distributed_tracking_descriptor", None)
    if descriptor is None:
        return ddp_callable(rank, *args)
    bound = inspect.signature(ddp_callable).bind(rank, *args)
    world_size = bound.arguments["world_size"]
    train_dataset = bound.arguments["train_dataset"]
    validation_dataset = bound.arguments["test_dataset"]
    batch_size = bound.arguments["batch_size"]
    local_rank = descriptor["local_rank"]
    if local_rank is None:
        local_rank = rank
    return run_tracked_distributed(
        harness,
        lambda: ddp_callable(rank, *args),
        descriptor["wandb_config"],
        rank=rank,
        local_rank=local_rank,
        model_family=descriptor["model_family"],
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        effective_batch_size=batch_size,
        world_size=world_size,
        initial_config=descriptor["initial_config"],
        backend=descriptor["backend"],
        event_directory=descriptor["event_directory"],
        variant="internal-ddp",
    )


def run_tracked_distributed(
    harness: Any,
    training_callable: Callable[[], Any],
    wandb_config: WandbConfig | None,
    *,
    rank: int,
    local_rank: int | None,
    model_family: Literal["atomic", "pairwise", "parameter"],
    train_dataset: Any,
    validation_dataset: Any,
    effective_batch_size: int,
    world_size: int,
    initial_config: Mapping[str, Any],
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
    variant: str = "external-ddp",
) -> Any:
    """Execute one DDP rank and independently clean up tracker and process group."""

    try:
        with harness_tracking(
            harness,
            wandb_config,
            model_family=model_family,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            effective_batch_size=effective_batch_size,
            world_size=world_size,
            initial_config=initial_config,
            backend=backend,
            event_directory=event_directory,
            is_primary=rank == 0,
            global_rank=rank,
            local_rank=local_rank,
            variant=variant,
        ):
            return training_callable()
    finally:
        # Many legacy loops clean up normally but not on exceptions.  Keep this
        # independent of tracker finalization and harmless after normal cleanup.
        primary_exception_active = sys.exc_info()[0] is not None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as exc:
            if primary_exception_active:
                print(
                    f"DDP cleanup failed while handling an error: {exc}",
                    file=sys.stderr,
                )
            else:
                raise


def current_harness_tracker(harness: Any) -> TrainingTracker:
    """Return the active tracker, or a no-op tracker for direct loop calls."""

    return getattr(harness, "_training_tracker", NullTrainingTracker())


_LOCAL_METRIC_VARIABLES = (
    ("total", "total_MAE_t", "total_MAE_v"),
    ("electrostatics", "elst_MAE_t", "elst_MAE_v"),
    ("exchange", "exch_MAE_t", "exch_MAE_v"),
    ("induction", "indu_MAE_t", "indu_MAE_v"),
    ("dispersion", "disp_MAE_t", "disp_MAE_v"),
    ("charge", "charge_MAE_t", "charge_MAE_v"),
    ("dipole", "dipole_MAE_t", "dipole_MAE_v"),
    ("quadrupole", "qpole_MAE_t", "qpole_MAE_v"),
    ("hfvr", "hfvr_MAE_t", "hfvr_MAE_v"),
    ("valence_width", "vw_MAE_t", "vw_MAE_v"),
)


def track_pretraining_from_locals(
    harness: Any,
    values: Mapping[str, Any],
    *,
    metric_labels: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> None:
    """Log an existing pre-training evaluation as epoch 0.

    ``values`` is the calling loop's ``locals()``; the conventional ``*_MAE_t`` /
    ``*_MAE_v`` names in :data:`_LOCAL_METRIC_VARIABLES` are picked up from it.
    The pre-training model is staged as the provisional best checkpoint because
    it is the only candidate that exists yet; the first epoch that the harness
    marks as an improvement replaces it.
    """

    _track_evaluation_boundary(
        harness,
        values,
        epoch=0,
        is_best=True,
        metric_labels=metric_labels,
        exclude=exclude,
    )


def track_epoch_from_locals(
    harness: Any,
    values: Mapping[str, Any],
    *,
    metric_labels: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> None:
    """Log one completed epoch from the calling loop's ``locals()``.

    ``epoch`` is read as the loop's zero-based counter and reported as
    ``epoch + 1``.  Whether the epoch is a new best is taken from the harness'
    own ``star_marker``/``test_lowered`` decision rather than recomputed, so the
    staged artifact always matches the checkpoint the harness itself kept.
    """

    marker = values.get("star_marker", values.get("test_lowered", False))
    _track_evaluation_boundary(
        harness,
        values,
        epoch=int(values.get("epoch", 0)) + 1,
        is_best=marker is True or marker == "*",
        metric_labels=metric_labels,
        exclude=exclude,
    )


def stage_final_weights(harness: Any) -> None:
    """Stage the final-epoch checkpoint before a loop restores its best weights.

    Several harness loops assign ``self.model = best_model`` before returning.
    Without this call ``harness_tracking`` would serialize those restored best
    weights and publish them under the ``final``/``latest`` aliases. Staging is
    skipped when the last epoch was also the best one, since the staged best
    checkpoint already holds the final weights.
    """

    state = getattr(harness, "_training_tracking_state", None)
    if state is None:
        # Tracking is disabled, or the loop was called outside ``harness_tracking``.
        return
    if state["best_epoch"] == state["epochs_completed"]:
        return
    state["final_path"] = stage_harness_checkpoint(
        current_harness_tracker(harness),
        harness,
        role="final",
        epoch=state["epochs_completed"],
        validation_loss=state["last_validation_loss"],
    )


def _track_evaluation_boundary(
    harness: Any,
    values: Mapping[str, Any],
    *,
    epoch: int,
    is_best: bool,
    metric_labels: Sequence[str] | None,
    exclude: Sequence[str],
) -> None:
    """Record one train/validation evaluation boundary for an active run."""

    state = getattr(harness, "_training_tracking_state", None)
    if state is None:
        # Tracking is disabled, or the loop was called outside ``harness_tracking``.
        return
    tracker = current_harness_tracker(harness)
    names, train_values, validation_values = _metrics_from_locals(
        values, metric_labels=metric_labels, exclude=exclude
    )
    if not names:
        return
    if not state["config_refreshed"]:
        tracker.update_config(_resolved_training_config(harness, {}))
        state["config_refreshed"] = True
    if state["defined_metrics"] is None:
        define_epoch_metrics(tracker, names)
        state["defined_metrics"] = tuple(names)
    elif tuple(names) != state["defined_metrics"]:
        raise ValueError("Training metric shape changed during a tracked run")
    validation_loss = values.get("test_loss")
    optimizer = values.get("optimizer")
    # Every tracked loop names its loaders `train_loader` / `test_loader`, so
    # the counts are read from the caller's locals rather than threaded as new
    # arguments through fourteen harness files.
    world_size = values.get("world_size")
    train_batches = _loader_batch_count(values.get("train_loader"), world_size)
    validation_batches = _loader_batch_count(
        values.get("test_loader"), world_size
    )
    if is_best:
        state["best_epoch"] = epoch
        state["best_validation_loss"] = validation_loss
        state["best_path"] = stage_harness_checkpoint(
            tracker,
            harness,
            role="best",
            epoch=epoch,
            validation_loss=validation_loss,
        )
    state["last_validation_loss"] = validation_loss
    state["epochs_completed"] = epoch
    log_epoch_metrics(
        tracker,
        epoch=epoch,
        metric_names=names,
        train_values=train_values,
        validation_values=validation_values,
        train_loss=values.get("train_loss"),
        validation_loss=validation_loss,
        train_batches=train_batches,
        validation_batches=validation_batches,
        learning_rate=(
            optimizer.param_groups[0]["lr"] if optimizer is not None else None
        ),
        epoch_seconds=values.get("dt"),
        is_best=is_best,
        # Read off the harness rather than threaded through fourteen call
        # sites, the same way the excluded-element list is. The guard above
        # compares only the MAE names, so extra keys are free to appear.
        extra_metrics=getattr(harness, "last_bound_occupancy", None),
    )


def _metrics_from_locals(
    values: Mapping[str, Any],
    *,
    metric_labels: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> tuple[list[str], list[Any], list[Any]]:
    """Collect conventional ``*_MAE_t``/``*_MAE_v`` locals into aligned lists.

    ``metric_labels`` renames the collected components, and is required when a
    harness' MAE local holds one value per predicted component rather than a
    single scalar.  ``exclude`` drops conventional names a model does not
    actually predict (for example dispersion under ``no_disp_nn``).
    """

    names: list[str] = []
    train_values: list[Any] = []
    validation_values: list[Any] = []
    for name, train_name, validation_name in _LOCAL_METRIC_VARIABLES:
        if name in exclude:
            continue
        if train_name not in values or validation_name not in values:
            continue
        train_value = values[train_name]
        validation_value = values[validation_name]
        width = _value_width(train_value)
        if metric_labels is None:
            if width != 1:
                raise ValueError(
                    f"Local {train_name!r} holds {width} components; pass "
                    "metric_labels naming each one"
                )
            labels = [name]
        else:
            labels = list(metric_labels)
            if len(labels) != width:
                raise ValueError(
                    f"metric_labels has {len(labels)} names for the {width} "
                    f"components in {train_name!r}"
                )
        if width == 1:
            names.append(labels[0])
            train_values.append(train_value)
            validation_values.append(validation_value)
        else:
            names.extend(labels)
            train_values.extend(train_value)
            validation_values.extend(validation_value)
    return names, train_values, validation_values


def _loader_batch_count(loader: Any, world_size: Any = None) -> int | None:
    """Batches one epoch draws from ``loader``, across every rank.

    Returns ``None`` whenever the count cannot be established -- an absent
    loader, or one without a length (an ``IterableDataset``). The caller then
    logs only the summed loss rather than inventing a denominator.

    Under DDP the epoch loops all-reduce their loss with ``ReduceOp.SUM``
    before it reaches tracking, so each rank contributes a sum over its own
    shard and the denominator spans every rank. ``world_size`` is read from the
    calling loop, where it is a parameter of the DDP entry points.
    """
    if loader is None:
        return None
    try:
        count = len(loader)
    except TypeError:
        return None
    if count <= 0:
        return None
    try:
        ranks = int(world_size) if world_size is not None else 1
    except (TypeError, ValueError):
        ranks = 1
    return count * max(ranks, 1)


def _value_width(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def log_epoch_metrics(
    tracker: TrainingTracker,
    *,
    epoch: int,
    metric_names: Sequence[str],
    train_values: Sequence[Any],
    validation_values: Sequence[Any],
    train_loss: Any | None = None,
    validation_loss: Any | None = None,
    train_batches: int | None = None,
    validation_batches: int | None = None,
    learning_rate: Any | None = None,
    epoch_seconds: Any | None = None,
    is_best: bool | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Log one evaluation boundary with stable train/validation metric names.

    Two loss metrics are emitted per split. ``*/loss_sum`` is the raw
    accumulator every epoch loop in the package returns: it adds each batch's
    already-averaged loss without dividing, so it is proportional to the batch
    count and cannot be compared between runs at different batch sizes. It is
    logged anyway because it is exactly the value the harnesses' own
    ``lowest_test_loss`` comparison uses, and the best-epoch decision has to
    stay auditable.

    ``*/loss_mean`` divides by the batch count, giving the mean per-batch
    objective -- the number the optimizer actually saw, on the same scale
    regardless of batch size. It is the unweighted mean of batch means, which
    differs from a true per-sample mean only by the short final batch.
    """

    if len(metric_names) != len(train_values) or len(metric_names) != len(
        validation_values
    ):
        raise ValueError("Metric names and train/validation values must have equal lengths")
    payload = epoch_metric_payload(
        epoch=epoch,
        learning_rate=learning_rate,
        epoch_seconds=epoch_seconds,
        is_best=is_best,
    )
    for prefix, loss, batches in (
        ("train", train_loss, train_batches),
        ("val", validation_loss, validation_batches),
    ):
        if loss is None:
            continue
        payload[f"{prefix}/loss_sum"] = scalar_value(
            loss, metric_name=f"{prefix}/loss_sum"
        )
        if batches:
            payload[f"{prefix}/loss_mean"] = scalar_value(
                loss / batches, metric_name=f"{prefix}/loss_mean"
            )
    for name, train_value, validation_value in zip(
        metric_names, train_values, validation_values
    ):
        train_key = f"train/mae/{name}"
        validation_key = f"val/mae/{name}"
        payload[train_key] = scalar_value(train_value, metric_name=train_key)
        payload[validation_key] = scalar_value(
            validation_value, metric_name=validation_key
        )
    for key, value in (extra_metrics or {}).items():
        payload[key] = scalar_value(value, metric_name=key)
    tracker.log(payload)


def define_epoch_metrics(
    tracker: TrainingTracker, metric_names: Sequence[str]
) -> None:
    """Define the exact common and model-specific epoch metric namespace."""

    names = [
        "train/loss_sum",
        "val/loss_sum",
        "train/loss_mean",
        "val/loss_mean",
    ]
    for name in metric_names:
        names.extend((f"train/mae/{name}", f"val/mae/{name}"))
    names.extend(
        (
            "optimizer/learning_rate",
            "timing/epoch_seconds",
            "checkpoint/is_best",
        )
    )
    tracker.define_metrics(names)


def _resolved_training_config(
    harness: Any, initial_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Add stable effective model facts without serializing model objects."""

    model = getattr(harness, "model", None)
    parameters = list(model.parameters()) if model is not None else []

    def count_parameters(*, trainable_only: bool) -> int | None:
        total = 0
        try:
            for parameter in parameters:
                if not trainable_only or parameter.requires_grad:
                    total += parameter.numel()
        except ValueError:
            # Lazy modules are initialized by the harness' existing warmup pass.
            return None
        return total

    config = {
        **dict(initial_config),
        "model/parameter_count_total": count_parameters(trainable_only=False),
        "model/parameter_count_trainable": count_parameters(trainable_only=True),
        "training/shuffle": getattr(
            harness, "train_shuffle", getattr(harness, "shuffle", None)
        ),
        "data/prebatched": getattr(harness, "prebatched", None),
    }
    get_config = getattr(model, "get_config", None)
    if callable(get_config):
        model_config = get_config()
        if isinstance(model_config, Mapping):
            for key, value in model_config.items():
                if _is_json_serializable(value):
                    config[f"model/{key}"] = value
    return {key: value for key, value in config.items() if value is not None}


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _model_config_from_attributes(model: Any) -> dict[str, Any]:
    """Build the same minimal architecture config used by legacy trainers."""

    attribute_names = (
        "n_message",
        "n_rbf",
        "n_neuron",
        "n_embed",
        "r_cut_im",
        "r_cut",
        "n_params",
        "param_start_mean",
        "param_start_std",
        "use_nn_screening",
        "precompute_hfvr",
        "use_atom_props",
        "use_precomputed_classical",
        "no_disp_nn",
    )
    config = {}
    for name in attribute_names:
        if hasattr(model, name):
            value = getattr(model, name)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if _is_json_serializable(value):
                config[name] = value
    return config


def _create_fallback_harness_checkpoint(
    harness: Any, *, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a loadable v2 checkpoint for legacy harnesses without a builder."""

    from . import model_io

    model = model_io.unwrap_model(harness.model)
    get_config = getattr(model, "get_config", None)
    config = get_config() if callable(get_config) else _model_config_from_attributes(model)

    # InducedDipoleModel's loader reconstructs its optional embedded modules from
    # these config entries.  Its state dict already contains their weights.
    if model.__class__.__name__ == "InducedDipoleMPNN":
        atomtype_model = getattr(model, "atomtype_hfvr_model", None)
        config.update(
            {
                "has_pretrained_atom_mpnn": getattr(model, "atom_mpnn_model", None)
                is not None,
                "atomtype_hfvr_config": (
                    _model_config_from_attributes(atomtype_model)
                    if atomtype_model is not None
                    else None
                ),
            }
        )

    submodels = {}
    for attribute_name in ("atom_model", "dimer_prop_model"):
        submodel = getattr(harness, attribute_name, None)
        if submodel is None or not hasattr(submodel, "state_dict"):
            continue
        submodel = model_io.unwrap_model(submodel)
        submodel_get_config = getattr(submodel, "get_config", None)
        submodel_config = (
            submodel_get_config()
            if callable(submodel_get_config)
            else _model_config_from_attributes(submodel)
        )
        submodels[attribute_name] = model_io.create_submodel_checkpoint(
            model=submodel,
            config=submodel_config,
            model_type=submodel.__class__.__name__,
        )

    return model_io.create_checkpoint(
        model=model,
        config=config,
        model_type=model.__class__.__name__,
        submodels=submodels or None,
        metadata=dict(metadata),
    )


def stage_harness_checkpoint(
    tracker: TrainingTracker,
    harness: Any,
    *,
    role: Literal["best", "final"],
    epoch: int,
    validation_loss: Any | None,
) -> Path | None:
    """Serialize a harness checkpoint into tracker-managed staging storage."""

    if not tracker.artifacts_enabled:
        return None
    from . import model_io

    metadata = {
        "checkpoint_role": role,
        "source_epoch": int(epoch),
        "validation_loss_sum": (
            scalar_value(validation_loss, metric_name="validation_loss_sum")
            if validation_loss is not None
            else None
        ),
    }
    create_checkpoint = getattr(harness, "_create_checkpoint", None)
    module_devices: list[tuple[Any, Any]] = []
    for attribute_name in ("model", "atom_model", "dimer_prop_model"):
        module = getattr(harness, attribute_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        try:
            device = next(module.parameters()).device
        except (StopIteration, ValueError):
            continue
        module_devices.append((module, device))
    try:
        if callable(create_checkpoint):
            checkpoint = create_checkpoint(metadata=metadata)
        else:
            checkpoint = _create_fallback_harness_checkpoint(
                harness, metadata=metadata
            )
    finally:
        # Several legacy checkpoint builders temporarily move modules to CPU.
        # Restore each rank's device before the next DDP collective/epoch.
        for module, device in module_devices:
            module.to(device)
    # Validate in memory: staging runs on the training hot path, so the extra
    # read-back of a freshly written checkpoint is not worth its cost per epoch.
    model_io.validate_checkpoint(checkpoint)
    path = tracker.staging_directory / f"{role}.pt"
    model_io.save_checkpoint(checkpoint, str(path))
    return path


def publish_harness_artifacts(
    tracker: TrainingTracker,
    *,
    best_path: Path | None,
    final_path: Path | None,
    best_epoch: int,
    best_validation_loss: Any | None,
) -> None:
    """Publish staged best/final checkpoints and update the run summary."""

    if not tracker.artifacts_enabled:
        return
    if best_path is None or final_path is None:
        raise RuntimeError("Enabled tracking requires staged best and final checkpoints")
    metadata = {
        "best_epoch": int(best_epoch),
        "best_validation_loss_sum": (
            scalar_value(best_validation_loss, metric_name="best_validation_loss_sum")
            if best_validation_loss is not None
            else None
        ),
    }
    if best_path == final_path:
        reference = tracker.log_checkpoint(
            final_path,
            aliases=("best", "final", "latest"),
            metadata={**metadata, "checkpoint_role": "best_and_final"},
        )
        best_reference = final_reference = reference
    else:
        best_reference = tracker.log_checkpoint(
            best_path,
            aliases=("best",),
            metadata={**metadata, "checkpoint_role": "best"},
        )
        final_reference = tracker.log_checkpoint(
            final_path,
            aliases=("final", "latest"),
            metadata={**metadata, "checkpoint_role": "final"},
        )
    summary: dict[str, Scalar | str | bool] = {
        "best/epoch": int(best_epoch),
        "checkpoint/best_artifact": best_reference,
        "checkpoint/final_artifact": final_reference,
    }
    if best_validation_loss is not None:
        summary["best/val_loss_sum"] = scalar_value(
            best_validation_loss, metric_name="best/val_loss_sum"
        )
    tracker.set_summary(summary)


def scalar_value(value: Any, *, metric_name: str) -> Scalar:
    """Convert a Python/numpy/torch scalar to a built-in scalar.

    NaN and infinity are passed through rather than rejected.  W&B stores them
    natively, and a diverged epoch must show up in the run instead of raising
    out of a tracking call and terminating training that would otherwise carry
    on (several harnesses handle NaN losses themselves).
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        result: Scalar = float(value)
    else:
        candidate = value
        if hasattr(candidate, "detach"):
            candidate = candidate.detach()
        if hasattr(candidate, "numel") and candidate.numel() != 1:
            raise ValueError(f"Metric {metric_name!r} must be scalar")
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        if isinstance(candidate, bool):
            return candidate
        if not isinstance(candidate, Real):
            raise TypeError(f"Metric {metric_name!r} is not numeric")
        result = float(candidate)
    return result


def epoch_metric_payload(
    *,
    epoch: int,
    learning_rate: Any | None = None,
    epoch_seconds: Any | None = None,
    is_best: bool | None = None,
) -> dict[str, Scalar]:
    """Build common epoch, optimizer, timing, and checkpoint metrics."""

    payload: dict[str, Scalar] = {"epoch": int(epoch)}
    optional = {
        "optimizer/learning_rate": learning_rate,
        "timing/epoch_seconds": epoch_seconds,
        "checkpoint/is_best": is_best,
    }
    for name, value in optional.items():
        if value is not None:
            payload[name] = scalar_value(value, metric_name=name)
    return payload


def _validated_metrics(metrics: Mapping[str, Any]) -> dict[str, Scalar]:
    if not isinstance(metrics, Mapping):
        raise TypeError("Metrics must be a mapping")
    result: dict[str, Scalar] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be non-empty strings")
        result[name] = scalar_value(value, metric_name=name)
    return result


def _validated_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    if not aliases:
        raise ValueError("At least one artifact alias is required")
    clean = tuple(dict.fromkeys(aliases))
    if not all(isinstance(alias, str) and alias for alias in clean):
        raise ValueError("Artifact aliases must be non-empty strings")
    return clean


def _ensure_json_serializable(value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("Tracking configuration must be JSON-serializable") from exc
