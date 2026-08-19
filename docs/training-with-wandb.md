# Training with Weights & Biases

QCMLForge can optionally record single-process and distributed training runs
in Weights & Biases (W&B). Tracking is disabled by default and W&B is not
imported by inference or by disabled training runs.

## Install

```bash
pip install -e '.[tracking]'
```

## CLI usage

Track an online run:

```bash
python train_models.py \
  --train_am AtomModel \
  --am_model_path ./models/am_example.pt \
  --wandb-mode online \
  --wandb-project qcmlforge \
  --wandb-group am-baseline \
  --wandb-tags atom baseline
```

Run without network access and sync later:

```bash
python train_models.py \
  --train_apnet APNet2 \
  --ap_model_path ./models/ap2_example.pt \
  --wandb-mode offline
wandb sync ./wandb/offline-run-*
```

The CLI also accepts `--wandb-entity`, `--wandb-name`, `--wandb-group`,
`--wandb-job-type`, `--wandb-notes`, and `--wandb-dir`. Standard `WANDB_*`
environment variables are used when the corresponding option is omitted.
`WANDB_MODE` activates tracking only when set to `online` or `offline`.

If one `train_models.py` invocation trains both an atomic and a pairwise model,
the models create independent runs in one group. An explicit run name is
suffixed with `-atom` and `-pairwise`.

## Direct Python API

Every public training harness accepts the same configuration object:

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

Passing no configuration, `None`, or `WandbConfig(mode="disabled")` preserves
the ordinary dependency-free training path.

## Metrics and checkpoints

Runs log one record per evaluation boundary. Existing pre-training evaluations
are logged as epoch 0; completed epochs use 1 through `n_epochs`. Metric names
use the `train/mae/*`, `val/mae/*`, `train/loss_sum`, and `val/loss_sum`
namespaces. `loss_sum` is intentionally not described as a sample-normalized
loss.

Each successful tracked run publishes the best-validation checkpoint with the
`best` alias and the final-epoch checkpoint with `final` and `latest`. When the
states are identical, one artifact version receives all three aliases. The
user-provided local `model_path` keeps its existing best-checkpoint semantics.
For example:

```python
import wandb

api = wandb.Api()
artifact = api.artifact(
    "ENTITY/PROJECT/qcmlforge-model-RUN_ID:best", type="model"
)
checkpoint_directory = artifact.download()
```

A checkpoint supplied to a harness is a **warm start**. Each call to `train()`
creates a new W&B run and records checkpoint lineage; it does not restore the
optimizer, scheduler, random-number state, split, best loss, or epoch counter,
and is not an exact W&B resume.

## Distributed training

Internal DDP launches retain the ordinary `model.train(world_size=N)` API and
create exactly one run owned by global rank 0. Tracker configuration and the
test backend descriptor are propagated through the spawn boundary; nonzero
ranks do not initialize tracking or stage artifacts. Epoch metrics are logged
after their distributed reductions.

`train_ddp_slurm.py` uses external DDP ownership: launch one script process per
SLURM task with `srun`, and each task enters its assigned worker directly rather
than creating a nested `mp.spawn` group. Global rank 0 is the only tracking
owner. This path has automated CPU process-level coverage, but has not been
validated under an actual multi-node SLURM allocation. Multi-node operation
remains experimental until a two-node SLURM smoke test is completed.
