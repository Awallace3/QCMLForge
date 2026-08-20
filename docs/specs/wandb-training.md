# Specification: Weights & Biases for All Model Training Runs

**Status:** Implemented  
**Scope:** `qcmlforge` / `apnet_pt` training harnesses  
**Decision date:** 2026-08-19

## 1. Summary

Add optional Weights & Biases (W&B) experiment tracking to every supported model-training path while preserving current behavior when tracking is not requested.

The integration will:

- be explicitly enabled; the default remains no telemetry and no W&B import;
- install through an optional dependency extra;
- work for both CLI entry points and direct calls to every public harness `.train()` method;
- initialize and log only from global rank 0 in distributed runs;
- record resolved configuration, available pre-training metrics, epoch metrics, best-model state, and final status;
- upload both best and final model checkpoints as versioned W&B artifacts;
- support online and offline W&B modes;
- treat a warm start from an existing checkpoint as a new run, not an exact resume.

This is an observability feature. It must not change model outputs, dataset splits, optimizer behavior, checkpoint compatibility, or stdout logging when disabled.

## 2. Product decisions

The following decisions are approved:

1. **Activation:** explicit opt-in.
2. **Dependency:** optional package extra, not a core inference dependency.
3. **Artifacts:** upload both the best-validation and final checkpoints.
4. **Naming:** project, entity, group, run name, and tags are configurable through CLI/API with `WANDB_*` environment fallbacks.
5. **Distributed logging:** one W&B run per training job; global rank 0 is the sole owner.
6. **Batch logging:** out of scope for v1. Log once per evaluation boundary to avoid excessive volume and synchronization.
7. **Gradient/parameter watching:** do not call `wandb.watch()` by default. It adds overhead and is fragile around `torch.compile` and DDP wrappers.

## 3. Goals

### 3.1 Functional goals

- Track all models reachable from `train_models.py`.
- Track `AtomInducedDipoleModel` runs launched by `train_ddp_slurm.py` after correcting its DDP launch ownership.
- Track direct Python API training for public harnesses not exposed by the main CLI.
- Use one stable metric namespace across model families.
- Preserve the existing best-checkpoint file and upload it as a W&B artifact.
- Serialize and upload the final in-memory model without changing which model is retained by each harness after training.
- Finish runs cleanly on success and mark them failed on exceptions.
- Produce useful local W&B data in offline mode with no network requirement.

### 3.2 Engineering goals

- Keep W&B-specific calls behind a small adapter.
- Make disabled tracking a no-op that does not require `wandb` to be installed.
- Make tracking testable with a fake backend and no W&B account or network.
- Avoid copying W&B lifecycle logic into every training loop.
- Add tracking incrementally without first rewriting all trainers into a common base class.

## 4. Non-goals

- Exact restoration of optimizer, scheduler, RNG, sampler, best-loss, or epoch state.
- W&B Sweeps, Launch, Reports, or automated hyperparameter optimization.
- Per-batch metrics, gradients, parameter histograms, or example molecule tables.
- One W&B run per DDP rank.
- Uploading entire datasets or raw molecular records.
- Replacing current stdout output or local checkpoints.
- Correcting unrelated DDP/SLURM architecture issues; launch ownership and metric-reducer defects that would make tracked runs invalid are in scope.
- Refactoring all model harnesses into a single trainer framework.

## 5. Current-state inventory

Training is implemented independently in multiple harnesses. Most follow:

```text
public train()
  -> create train/test split
  -> single_proc_train() OR mp.spawn(ddp_train())
  -> pre-training evaluation
  -> repeated train/evaluate epochs
  -> save best checkpoint
  -> print epoch metrics
```

There is no shared callback or event layer. Therefore, a CLI-only integration would miss direct API calls and would not have access to epoch metrics.

### 5.1 Required harness coverage

| File | Harness | Metrics/profile |
|---|---|---|
| `src/apnet_pt/AtomModels/ap2_atom_model.py` | `AtomModel` | charge, dipole, quadrupole |
| `src/apnet_pt/AtomModels/ap2_hirshfeld_atom_model.py` | `AtomHirshfeldModel` | atomic metrics plus HFVR and valence width |
| `src/apnet_pt/AtomModels/ap3_atom_model.py` | `AtomInducedDipoleModel` | charge, dipole, quadrupole |
| `src/apnet_pt/AtomModels/ap3_atom_model_frozen.py` | `InducedDipoleModel` | charge, dipole, quadrupole |
| `src/apnet_pt/AtomModels/ap3_atomtype_mpnn.py` | `AtomTypeParamModel` | HFVR and valence width |
| `src/apnet_pt/AtomPairwiseModels/apnet2.py` | `APNet2Model` | total and SAPT components |
| `src/apnet_pt/AtomPairwiseModels/apnet2_fused.py` | `APNet2_AM_Model` | total and SAPT components |
| `src/apnet_pt/AtomPairwiseModels/apnet3.py` | `APNet3Model` | total and SAPT components |
| `src/apnet_pt/AtomPairwiseModels/apnet3_fused.py` | `APNet3_AtomType_Model` | standard and FSAPT branches |
| `src/apnet_pt/AtomPairwiseModels/apnet3_d3_fused.py` | `APNet3D3_AtomType_Model` | 3- or 4-component training |
| `src/apnet_pt/AtomPairwiseModels/apnet3_fused_variants.py` | `APNet3_AtomType_Model` variant | total and SAPT components |
| `src/apnet_pt/AtomPairwiseModels/dapnet2.py` | `APNet2_dAPNet2Model`, `dAPNet2Model` | scalar or component pairwise metrics |
| `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py` | `AM_DimerParam_Model`, pairwise `AtomTypeParamModel` | dynamically selected parameter targets |

“All model training runs” means all rows above, including callable legacy harnesses that are not selected by `train_models.py`.

### 5.2 Entry points

- `train_models.py` can run atom and pairwise training sequentially in one process. These must become two separate W&B runs, optionally sharing a group.
- `train_ddp_slurm.py` is a separate launcher for `AtomInducedDipoleModel`.
- Shell launchers inherit behavior from their Python entry point.
- Direct `.train()` calls are supported and must not require the CLI.

### 5.3 Current checkpoint semantics

Current training saves the model with the lowest validation loss. Loading an existing path restores model configuration and weights but creates a new optimizer, scheduler, split, and epoch counter. This is a **warm start**, not an exact resume.

W&B run resumption must not claim stronger semantics than the trainer provides. In v1:

- every `.train()` invocation creates a new W&B run;
- W&B SDK `resume` is set to `"never"`;
- a warm-start checkpoint is recorded as lineage metadata and may be attached with `run.use_artifact()` only when it is already a W&B artifact;
- every harness that accepts or loads a pretrained checkpoint retains sanitized source-checkpoint metadata on the harness so a later direct `.train()` call can report lineage;
- exact run resume is deferred to a future training-state checkpoint specification.

## 6. Public API

### 6.1 Optional dependency

Add an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
tracking = ["wandb>=0.19.9,<1"]
```

The lower bound supports current distributed logging behavior documented by W&B. The upper bound prevents an unreviewed major-version upgrade.

Installation:

```bash
pip install -e '.[tracking]'
```

No module imported by inference-only workflows may import `wandb` eagerly.

### 6.2 Configuration type

Add `src/apnet_pt/training_tracking.py` with an immutable, pickle-safe configuration object:

```python
@dataclass(frozen=True)
class WandbConfig:
    mode: Literal["disabled", "online", "offline"] = "disabled"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    job_type: str = "train"
    notes: str | None = None
    directory: str | None = None
```

Requirements:

- `mode="disabled"` is the default and never imports W&B.
- `project=None` resolves from `WANDB_PROJECT`, then `"qcmlforge"`.
- `entity=None`, `group=None`, and `name=None` resolve from the corresponding W&B environment variables when present.
- Tags supplied by API/CLI are merged with generated model-family and launch-mode tags.
- The object contains only primitive/pickle-safe values so it can cross `mp.spawn` boundaries.
- Unknown fields are rejected rather than silently ignored.

Every public harness `.train()` gains a keyword-only-compatible argument:

```python
wandb_config: WandbConfig | None = None
```

`None` is equivalent to `WandbConfig(mode="disabled")`.

This exact argument must be added to every harness before `train_models.py` forwards it. This avoids the existing `inspect.signature()` filtering silently dropping telemetry for individual variants.

### 6.3 CLI

Add the same options to `train_models.py` and `train_ddp_slurm.py`:

```text
--wandb-mode {disabled,online,offline}
--wandb-project PROJECT
--wandb-entity ENTITY
--wandb-name NAME
--wandb-group GROUP
--wandb-tags TAG [TAG ...]
--wandb-job-type JOB_TYPE
--wandb-notes NOTES
--wandb-dir PATH
```

Resolution rules:

1. explicit CLI/API value;
2. standard `WANDB_*` environment value;
3. repository default.

`--wandb-mode` defaults to `WANDB_MODE` when it is `online`, `offline`, or `disabled`; otherwise it defaults to `disabled`. Therefore activation remains explicit through either CLI or environment.

When one `train_models.py` invocation trains both atom and pairwise models:

- create two independent runs;
- use the requested group for both;
- if no group is supplied, generate one invocation ID and use it for both;
- suffix an explicit run name with `-atom` and `-pairwise`; otherwise generate names from harness class and output checkpoint stem.

## 7. Tracking architecture

### 7.1 Adapter

`training_tracking.py` will expose a W&B-neutral adapter used by harnesses:

```python
class TrainingTracker(Protocol):
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
```

Implementations:

- `NullTrainingTracker`: no-op, dependency-free.
- `WandbTrainingTracker`: lazy-imports `wandb` only in `start()`.
- `FileEventTrainingTracker`: test backend that writes append-only JSONL events to a supplied directory and is safe to construct independently in spawned children.

`log_checkpoint()` attaches all aliases to one artifact version atomically and returns the qualified artifact reference stored in the run summary. `staging_directory` is managed by the tracker and exists in online and offline modes; harnesses may serialize temporary best/final checkpoints there.

Factory:

```python
def create_training_tracker(
    wandb_config: WandbConfig | None,
    *,
    is_primary: bool,
    run_context: RunContext,
    backend: TrackerBackend = TrackerBackend.WANDB,
    event_directory: str | None = None,
) -> TrainingTracker:
    ...
```

The factory returns the null tracker when disabled or when `is_primary` is false. `NullTrainingTracker.artifacts_enabled` is always false; the W&B/file-event trackers return true only after successful startup in the primary process. `TrackerBackend` and `event_directory` are primitive/pickle-safe descriptors propagated through `mp.spawn`; production public APIs use the W&B backend, while tests may select the file-event backend. Tests must never rely on a parent-only monkeypatch or in-memory list to observe spawned children.

Every public harness `.train()` also accepts private keyword-only `_tracker_backend: TrackerBackend = TrackerBackend.WANDB` and `_tracker_event_directory: str | None = None` plumbing arguments. They are not documented as user API and exist solely for repository tests and internal launch adapters. The parent forwards them unchanged to direct or spawned epoch-loop workers, which pass them to `create_training_tracker()`. Production CLIs never expose these arguments and always select the W&B backend.

### 7.2 Run context

A serializable `RunContext` carries resolved facts known before the epoch loop:

- harness and low-level model class names;
- model family (`atomic`, `pairwise`, `parameter`);
- training variant/mode, including transfer learning, FSAPT, and no-dispersion mode;
- requested output path;
- warm-start source path, if any;
- world size, global rank, local rank when known, and device type;
- dataset class, dataset specification/type, train size, validation size, and effective batch size;
- Git commit and dirty state when available;
- `qcmlforge`, PyTorch, CUDA, Python, and platform versions.

Do not record credentials, full molecule contents, database tokens, or unrestricted environment dumps. Paths should be normalized to user-approved paths or basenames; avoid exposing private absolute paths by default.

### 7.3 Lifecycle ownership

The actual epoch-loop process owns the run. The adapter implements an explicit `NEW -> STARTED -> FINISHED` state machine: `finish()` is idempotent, does nothing before a successful `start()`, and never raises while another exception is already active.

```python
tracker = create_training_tracker(
    wandb_config,
    is_primary=(rank == 0),
    run_context=context,
)
completed = False
try:
    tracker.start(config=initial_config)
    tracker.update_config(resolved_config)
    ...
except BaseException:
    if tracker.started:
        tracker.set_summary_safely({"run/status": "failed"})
    raise
else:
    tracker.set_summary({"run/status": "completed"})
    completed = True
finally:
    tracker.finish_safely(exit_code=0 if completed else 1)
    cleanup_distributed_state_safely()
```

`*_safely` methods report secondary tracker failures to stderr and preserve any active primary exception. When no primary exception is active, they raise tracker/finish failures so an otherwise successful enabled run cannot silently lose telemetry. Distributed cleanup is performed in `finally` independently of tracker state and follows the same primary-exception precedence.

Rules:

- Do not initialize W&B in the parent before `mp.spawn`.
- In single-process runs, rank 0 owns the run.
- In internal DDP runs, spawned global rank 0 owns the run.
- Nonzero ranks receive a null tracker and make no W&B SDK calls.
- Metrics must be globally reduced before rank 0 logs them.
- The tracker must finish exactly once.
- If tracker initialization fails because `wandb` is missing, raise an actionable `ImportError` explaining `pip install 'qcmlforge[tracking]'`.
- Missing-package and configuration errors are validated by the CLIs before model/dataset construction when possible. Online initialization occurs before compilation, loader iteration, and epoch work, but may follow model/dataset construction and splitting; the integration does not promise to preflight W&B authentication before dataset processing. Users requiring no network must select offline mode.

### 7.4 SLURM launcher constraint

`train_ddp_slurm.py` currently executes the public `.train(world_size > 1)` call in every externally launched rank, while that method internally calls `mp.spawn`. This can produce nested/duplicated training jobs independently of W&B.

Correct launch ownership is a v1 prerequisite, not merely a tracking guard. Implement the external-DDP model for this SLURM-specific entry point: each SLURM task enters one rank-aware `AtomInducedDipoleModel` worker path and must not call another `mp.spawn`. The internal-spawn `.train()` path remains available to non-SLURM single-node callers.

The external worker must receive global rank, local rank, world size, rendezvous configuration, and `wandb_config`; only global rank 0 owns W&B. Tests must prove that an N-task launch creates one N-rank training group and one W&B lifecycle, not N competing N-rank groups. Multi-node support may be documented only after a two-node SLURM smoke test; until then the launcher documentation must state single-node validation only.

## 8. Configuration recorded for each run

The run config must contain resolved, effective values rather than only raw CLI arguments.

Required common fields:

```text
model/harness_class
model/module_class
model/family
model/parameter_count_total
model/parameter_count_trainable
training/epochs
training/effective_batch_size
training/learning_rate_initial
training/learning_rate_decay
training/learning_rate_final
training/split_percent
training/shuffle
training/random_seed
training/transfer_learning
training/skip_compile
training/include_total_mse
training/world_size
data/dataset_class
data/spec_type
data/dataset_type
data/train_samples
data/val_samples
data/prebatched
checkpoint/output_name
checkpoint/warm_start
runtime/device
runtime/torch_version
runtime/cuda_version
runtime/qcmlforge_version
```

Add model-specific architecture fields from each model's existing `get_config()` or checkpoint config. Values must be JSON-serializable. Large tensors, state dicts, dataset objects, and callables must not enter W&B config.

Where the requested and effective value differ, record both, for example:

```text
training/batch_size_requested
training/effective_batch_size
```

## 9. Metric schema

### 9.1 Step semantics

- `epoch=0` is logged when the harness already performs a pre-training evaluation or when an isolated evaluation can be added without changing training RNG/state.
- Completed training epochs are logged as `epoch=1..n_epochs`.
- One `tracker.log()` call contains all metrics for an evaluation boundary.
- W&B metrics use `epoch` as their explicit step metric.
- Existing console epoch numbering does not need to change.

Define metrics at startup:

```python
run.define_metric("epoch")
run.define_metric("train/*", step_metric="epoch")
run.define_metric("val/*", step_metric="epoch")
run.define_metric("optimizer/*", step_metric="epoch")
run.define_metric("timing/*", step_metric="epoch")
run.define_metric("val/loss_sum", summary="min")
run.define_metric("val/mae/total", summary="min")
```

Only define model-specific metrics that the harness actually emits.

### 9.2 Common metrics

```text
epoch
train/loss_sum
val/loss_sum
optimizer/learning_rate
timing/epoch_seconds
checkpoint/is_best
```

Current `total_loss` values are often accumulated sums of per-batch means and are not consistently normalized by sample count. The v1 metric name is therefore `loss_sum`, not `loss`. Cross-run comparisons are valid only when batch construction and dataset size are comparable. Standardizing normalized loss is separate work.

Before semantic MAE keys are emitted, every corresponding reducer must be numerically correct. Existing pairwise DDP paths that divide per-sample component numerators by `preds.numel()`, omit dispersion all-reduction, or define total MAE as mean component error must be corrected. Atomic epoch-0 values must also be globally reduced before logging. These corrections require world-size-1 versus world-size-2 numerical equivalence tests for each reducer shape; until a family passes, tracking for that family must not advertise the semantic key.

### 9.3 Atomic metrics

```text
train/mae/charge
val/mae/charge
train/mae/dipole
val/mae/dipole
train/mae/quadrupole
val/mae/quadrupole
train/mae/hfvr
val/mae/hfvr
train/mae/valence_width
val/mae/valence_width
```

Each harness logs only its supported subset. Metric descriptions must document the same averaging and units used by its current batch/evaluation implementation; implementation must not infer or relabel unverified units.

### 9.4 Pairwise metrics

```text
train/mae/total
val/mae/total
train/mae/electrostatics
val/mae/electrostatics
train/mae/exchange
val/mae/exchange
train/mae/induction
val/mae/induction
train/mae/dispersion
val/mae/dispersion
```

- AP3-D3 with `no_disp_nn=True` omits dispersion training metrics.
- Transfer-learning/scalar branches emit total only.
- FSAPT v1 logs only the aggregate total and SAPT-component MAEs currently returned by its evaluator. Per-fragment metrics are deferred until evaluators expose authoritative fragment labels and values.
- `AM_DimerParam_Model` derives names from the selected target labels (`y_ind`/configured property names) and validates label/value lengths before logging.
- Pairwise MAEs use the training target's existing energy units; expected kcal/mol usage must be confirmed in tests/docs rather than converted by the tracker.

### 9.5 Run summary

At completion record:

```text
run/status = completed | failed
run/epochs_completed
best/epoch
best/val_loss_sum
best/val_mae_total (when available)
checkpoint/best_artifact
checkpoint/final_artifact
runtime/total_seconds
```

On failure also record the exception class and a bounded, sanitized message. Do not upload a full environment dump or secrets from tracebacks.

## 10. Model artifacts

### 10.1 Artifact identity

Use one artifact collection per run:

```text
name: qcmlforge-model-<wandb_run_id>
type: model
```

Log checkpoint versions with aliases:

- best checkpoint: `best`;
- final checkpoint: `final` and `latest`.

If final and best are byte-identical, log one version and attach all three aliases.

Artifact metadata:

```text
harness_class
model_class
checkpoint_format_version
checkpoint_role
source_epoch
validation_loss_sum
wandb_run_id
git_commit
warm_start_source_name
```

### 10.2 Best checkpoint

The local `model_path` remains the best-validation checkpoint. Do not upload a version on every improvement. At training completion:

- only when `tracker.artifacts_enabled` is true, snapshot the epoch-0 candidate before optimization into the tracker staging directory, even when `model_path` is `None`;
- treat the pre-training evaluation as the initial best candidate and log `checkpoint/is_best=true` at epoch 0, since it is the only candidate that exists yet; harnesses that seed their own `lowest_test_loss` with infinity replace it at the first evaluated epoch, and harnesses that seed it from the pre-training loss keep it until a real improvement;
- replace the staged best snapshot whenever the harness' own improvement flag is set, reusing that decision rather than recomputing it;
- validate each staged checkpoint in memory before writing it, without reading it back, because staging runs on the epoch hot path;
- upload the final best snapshot once with alias `best`;
- support `n_epochs=0` and no-improvement runs with and without `model_path`;
- preserve the existing user-visible `model_path` semantics and do not delete or rename it.

### 10.3 Final checkpoint

The final checkpoint represents model state immediately after the final epoch, before any harness-specific restoration of an in-memory best model. Each harness must use its existing checkpoint construction logic so embedded atom/dimer submodels and configuration remain loadable.

When `tracker.artifacts_enabled` is true, write the final checkpoint into the tracker staging directory once at the end of the run, upload with aliases `final` and `latest`, and allow W&B/offline storage to retain it. Do not serialize a final checkpoint on every epoch. When the last completed epoch was also the best one the staged best checkpoint already holds the final weights, so no second file is written and the single version carries all three aliases. Disabled and nonprimary processes must not perform staging serialization. Do not create an extra sibling checkpoint beside the user's `model_path` unless a future CLI option explicitly requests one.

For harnesses that restore `best_model` at the end, capture the final-epoch checkpoint before that restoration. The local/in-memory post-training behavior remains unchanged.

### 10.4 Artifact failure policy

- Failure to serialize a checkpoint is a training failure because the artifact contract is explicit when W&B is enabled.
- Metric values are never a training failure. A NaN or infinite loss is logged as-is; tracking must not terminate a run that would otherwise continue.
- A missing `model_path` does not disable artifacts; best and final checkpoints are serialized into the run directory.
- Disabled tracking does not create W&B artifact files or alter checkpoint behavior.

## 11. Harness integration

### 11.1 Shared event helpers

Every training loop reports itself through two explicit calls rather than having
its methods wrapped from outside:

```python
track_pretraining_from_locals(self, locals(), metric_labels=..., exclude=...)
track_epoch_from_locals(self, locals(), metric_labels=..., exclude=...)
```

Both read the conventional `train_loss`, `test_loss`, `epoch`, `dt`,
`optimizer`, `star_marker`/`test_lowered`, and `<name>_MAE_t`/`<name>_MAE_v`
locals the loops already define, so the loops keep their existing shape.
`metric_labels` names the components of an MAE local that holds one value per
predicted term; `exclude` drops a conventional metric a model does not predict
(dispersion under `no_disp_nn`).

The helpers:

- accept Python numbers or scalar tensors;
- detach tensors and convert them to Python scalars, passing NaN/infinity
  through so a diverged epoch is visible in the run instead of raising into
  training;
- reject non-scalar or non-numeric values with a clear metric name;
- omit optional metrics the loop did not compute;
- never perform distributed collectives;
- no-op when tracking is disabled or the caller is not the tracking owner.

No harness method is monkeypatched, in production or in tests.

### 11.2 Per-loop changes

For every `single_proc_train()` and `ddp_train()` listed in Section 5:

1. initialize the tracker after rank/device and initial model context are known but before compilation, loader iteration, or epoch work;
2. update tracker config after splitting/loaders reveal effective sizes and batch settings;
3. when `tracker.artifacts_enabled` is true, snapshot the epoch-0 model as the initial best candidate; disabled and nonprimary ranks skip all tracker-driven serialization;
4. log pre-training metrics at epoch 0 only when already available or computed through a nonshuffled evaluation loader with dedicated generator and preserved CPU/CUDA/dataloader RNG state;
5. after each globally reduced evaluation, log the full epoch payload;
6. mark `checkpoint/is_best` from the harness' own `star_marker`/`test_lowered` flag, which is the same comparison that drives checkpoint saving;
7. capture best checkpoint metadata at the successful save boundary;
8. capture the final-epoch checkpoint before restoring any best model;
9. upload best/final artifacts and finish the run;
10. finish with failure status, preserve the primary exception, and clean up distributed state.

Existing `print()` calls remain.

### 11.3 Public `.train()` changes

Each `.train()` method:

- accepts `wandb_config` plus the private test-only `_tracker_backend` and `_tracker_event_directory` keyword arguments;
- performs dataset splitting exactly as before;
- adds effective dataset/model fields to `RunContext`;
- passes the serializable config/context and private tracker-runtime descriptor into `mp.spawn` workers or the direct loop;
- never starts a run itself unless it is also the actual epoch-loop process.

## 12. Implementation files

### New files

- `src/apnet_pt/training_tracking.py`
- `tests/test_training_tracking.py`
- `docs/wandb.md`

### Modified files

- `pyproject.toml`
- `train_models.py`
- `train_ddp_slurm.py`
- every harness file listed in Section 5.1
- focused existing training tests where fixture reuse is useful
- `README.md` with a short link to the W&B guide

Do not place core tracking code in either CLI; direct API use must remain fully supported.

## 13. Test plan

All automated tests use the file-event tracker backend. Unit and integration tests must not contact W&B or require authentication. Spawn tests pass a pickle-safe backend descriptor and assert process-safe JSONL events written by child processes.

### 13.1 Configuration and dependency tests

- Default/`None` config returns `NullTrainingTracker` without importing `wandb`.
- Online/offline config without the extra raises the documented installation error.
- API values override environment values.
- Invalid mode, tags, or nonserializable config values fail clearly.
- CLI parsing and forwarding work in both entry points.
- One CLI invocation with atom and pairwise training creates two lifecycle records in one group.

### 13.2 Metric-helper tests

Test exact payloads for:

- charge/dipole/quadrupole atomic output;
- Hirshfeld/HFVR/valence-width output;
- four-component pairwise output;
- AP3-D3 three-component output;
- scalar transfer-learning/dAPNet output;
- FSAPT output;
- dynamically labeled `AM_DimerParam_Model` output;
- scalar tensor conversion and rejection of NaN/Inf/non-scalars.

### 13.3 Representative loop tests

Use tiny existing datasets and 1-2 epochs. Add:

- a parameterized signature/forwarding test for every harness in Section 5.1;
- an event and loadable-artifact test for every distinct loop/checkpoint-construction family;
- AP3-D3 `no_disp_nn` coverage;
- AP3 fused FSAPT coverage;
- `AM_DimerParam_Model` dynamic-metric coverage;
- coverage for every harness that restores best model after training;
- `n_epochs=0` and no-improvement cases for atomic and pairwise families.

Assert:

- one start and one finish;
- one event per completed epoch, plus epoch 0 where the harness supplies a state-preserving pre-training evaluation;
- config contains effective split and batch size;
- best marker matches current checkpoint comparison;
- best and final artifacts are loadable with current model I/O;
- disabled mode leaves existing checkpoint and output behavior unchanged.

### 13.4 Distributed tests

Without network access:

- rank 0 initializes/logs/uploads/finishes once;
- nonzero ranks use the null tracker;
- logging happens after global metric reduction;
- world-size-1 and world-size-2 MAEs agree numerically for total, every component including dispersion, atomic metrics, and three-component mode;
- serializable configuration survives `mp.spawn`;
- the supported SLURM launch creates one run, not one per process;
- DDP-wrapped/compiled checkpoints remain compatible with `model_io.unwrap_model()`.

At least one world-size-2 CPU test should exercise the real spawn path when CI resources permit; otherwise mark it as a dedicated integration job rather than silently skipping all DDP coverage.

### 13.5 Failure tests

- Exception during training produces `finish(exit_code=1)` exactly once and re-raises.
- Tracker initialization failure occurs before epoch 1.
- Artifact serialization failure is surfaced.
- Offline mode requires no API key or network.
- A warm start through CLI or direct API creates a new run and records lineage; it does not use W&B resume.
- Enabled and disabled fixed-seed runs produce equivalent final weights, including harnesses where epoch-0 evaluation was newly added.

## 14. Acceptance criteria

The feature is complete when:

1. `pip install -e .` supports all existing inference/training behavior with W&B absent.
2. `pip install -e '.[tracking]'` enables online and offline tracking.
3. Tracking is disabled by default for CLI and direct API calls.
4. Every harness in Section 5.1 accepts the same `wandb_config` API.
5. Every harness emits per-epoch metrics with the schema in Section 9 and emits pre-training metrics where Section 9.1 permits epoch 0.
6. Exactly one W&B run is created per training invocation, including internal-spawn DDP and the corrected external-SLURM worker path.
7. Sequential atom/pairwise training creates two grouped runs.
8. The best and final checkpoints are uploaded and loadable through existing checkpoint APIs.
9. Existing best-checkpoint paths and post-training in-memory model behavior are unchanged when tracking is disabled.
10. Exceptions close the W&B run as failed and preserve the original exception.
11. Tests perform no real W&B network calls.
12. Every metric reducer used for semantic W&B keys passes world-size-1/world-size-2 numerical equivalence tests.
13. Enabled tracking does not perturb fixed-seed training results.
14. Documentation includes online, offline, direct API, DDP, artifact retrieval, and warm-start examples.
15. The full existing test suite passes.

## 15. Rollout plan

### Phase 1: adapter and canonical harnesses

- Add optional dependency, config, null/fake/W&B adapters, metric helpers, and tests.
- Integrate `AtomModel` and `APNet2Model` single-process loops.
- Validate online with one disposable development project and offline in CI/manual testing.

### Phase 2: all single-process harnesses

- Propagate the API and event calls through every required harness.
- Add model-family metric and artifact coverage.
- Update the main CLI and user documentation.

### Phase 3: DDP and SLURM

- Integrate every internal DDP loop with rank-zero ownership.
- Add CPU world-size-2 tests.
- Replace nested SLURM spawning with the external-DDP worker path and one global-rank-0 telemetry owner.
- Correct and validate duplicated DDP metric reducers before enabling their semantic metric keys.
- Do not claim multi-node support until a two-node smoke test passes.

### Phase 4: hardening

- Run full regression tests.
- Compare fixed-seed disabled-before/after and enabled-versus-disabled runs for equivalent checkpoints and metrics.
- Verify offline sync and best/final artifact retrieval.
- Review path/config redaction and failure messages.

## 16. Documentation examples to provide

CLI online:

```bash
python train_models.py \
  --train_am AtomModel \
  --am_model_path ./models/am_example.pt \
  --wandb-mode online \
  --wandb-project qcmlforge \
  --wandb-group am-baseline
```

CLI offline:

```bash
python train_models.py \
  --train_apnet APNet2 \
  --ap_model_path ./models/ap2_example.pt \
  --wandb-mode offline
wandb sync ./wandb/offline-run-*
```

Direct API:

```python
from apnet_pt.training_tracking import WandbConfig

model.train(
    dataset=dataset,
    n_epochs=50,
    model_path="./models/model.pt",
    wandb_config=WandbConfig(
        mode="online",
        project="qcmlforge",
        group="apnet2-ablation",
        tags=("apnet2", "seed-42"),
    ),
)
```

Warm starts must be described as new runs linked to a source checkpoint, not as exact resumes.

## 17. Deferred follow-ups

- Exact training resume with optimizer/scheduler/RNG/sampler/global-step state.
- Optional normalized, sample-weighted loss metrics across all harnesses.
- Per-batch logging with explicit cost controls.
- W&B Sweeps configuration generation.
- Dataset artifacts containing manifests/hashes but no raw private records.
- True multi-node external DDP support in `train_ddp_slurm.py`.

## 18. References

Official W&B documentation used for lifecycle and schema decisions:

- Run initialization: <https://docs.wandb.ai/models/ref/python/functions/init>
- Offline and disabled modes: <https://docs.wandb.ai/support/models/articles/what-is-the-difference-between-wandbinit>
- Resuming runs: <https://docs.wandb.ai/models/runs/resuming>
- Distributed training: <https://docs.wandb.ai/models/track/log/distributed-training>
- Metric axes and `define_metric`: <https://docs.wandb.ai/models/track/log/customize-logging-axes>
- Run summaries: <https://docs.wandb.ai/models/track/log/log-summary>
- Artifacts: <https://docs.wandb.ai/models/artifacts>
- Custom artifact aliases: <https://docs.wandb.ai/models/artifacts/create-a-custom-alias>
