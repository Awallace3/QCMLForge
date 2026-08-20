# Weights & Biases

## Log a training run

Install and authenticate once:

```bash
python -m pip install wandb
wandb login
```

Run the AP3 atom-model logging test used during development:

```bash
RUN="ap3-atom-pytest-100ep-$(date +%Y%m%d-%H%M%S)"
CHECKPOINT="$PWD/models/wandb/$RUN.pt"
WANDB_DIR="$PWD/wandb-runs/$RUN"

mkdir -p "$(dirname "$CHECKPOINT")" "$WANDB_DIR"

python -u train_models.py \
  --train_am AtomInducedDipoleModel \
  --am_model_path "$CHECKPOINT" \
  --atom_type_param_model_path \
    "$PWD/tests/test_models/ap3_ensemble_0/atp_mpnn_1.pt" \
  --data_dir "$PWD/tests/test_data_path" \
  --spec_type_am 6 \
  --n_epochs_atom 100 \
  --omp_num_threads 4 \
  --random_seed 42 \
  --wandb-mode online \
  --wandb-project qcmlforge \
  --wandb-name "$RUN" \
  --wandb-job-type test \
  --wandb-tags logging-test pytest-data ap3 \
  --wandb-dir "$WANDB_DIR"
```

Use absolute checkpoint and W&B paths. A successful run uploads epoch metrics and
model artifacts with these aliases:

- `best`: lowest validation loss
- `final`: final epoch
- `latest`: same artifact as `final`

Useful optional flags:

```text
--wandb-entity ENTITY
--wandb-group GROUP
--wandb-notes "DESCRIPTION"
--wandb-mode offline
```

For unattended jobs, set `WANDB_API_KEY` in the job environment; do not put it in
the command or repository.

## Download a model and run inference

Use `:best` while selecting models. Pin the resolved immutable version, such as
`:v0`, in production.

```python
from pathlib import Path

import qcelemental as qcel
import torch
import wandb

from apnet_pt import AtomModels

ARTIFACT = (
    "<YOUR_WANDB_PROJECT>/"
    "qcmlforge/qcmlforge-model-p6j5m3ei:v0"
)
CACHE = Path("wandb-models/ap3-p6j5m3ei")
HFVR_CHECKPOINT = Path("tests/test_models/ap3_ensemble_0/atp_mpnn_1.pt")

artifact = wandb.Api().artifact(ARTIFACT, type="model")
artifact_dir = Path(artifact.download(root=str(CACHE)))
checkpoint = artifact_dir / "best.pt"

model = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
    pre_trained_model_path=str(checkpoint),
    atomtype_hfvr_pre_trained_path=str(HFVR_CHECKPOINT),
    ignore_database_null=True,
    use_GPU=torch.cuda.is_available(),
)

mol = qcel.models.Molecule.from_data(
    """
    0 1
    O  0.000000  0.000000  0.000000
    H  0.758602  0.000000  0.504284
    H -0.758602  0.000000  0.504284
    units angstrom
    """
)

# One tuple per molecule: (charges, dipoles, quadrupoles, hidden states).
predictions = model.predict_qcel_mols([mol], batch_size=1)
charges, dipoles, quadrupoles, hidden_states = predictions[0]
```

The current AP3 constructor still requires the AtomTypeParam/HFVR checkpoint to
instantiate the embedded HFVR network. Keep that dependency versioned with the
inference deployment.

## Inspect runs and artifacts

```python
import wandb

api = wandb.Api()
run = api.run(
    "<YOUR_WANDB_PROJECT>/qcmlforge/p6j5m3ei"
)

print(run.url, run.state)
print(run.summary["best/epoch"])
print(run.summary["best/val_loss_sum"])

for artifact in run.logged_artifacts():
    print(artifact.qualified_name, list(artifact.aliases))
```

## Operational notes

Install QCMLForge with the optional tracking dependency when developing locally:

```bash
python -m pip install -e '.[tracking]'
```

Tracking is disabled by default. Omitting `wandb_config`, passing `None`, or
using `WandbConfig(mode="disabled")` does not import W&B. CLI options override
standard `WANDB_*` environment variables. To capture offline runs and sync them
later:

```bash
WANDB_MODE=offline python -u train_models.py [TRAINING OPTIONS]
wandb sync ./wandb/offline-run-*
```

When one `train_models.py` command trains both atomic and pairwise models, it
creates two W&B runs in one group. An explicit name receives `-atom` and
`-pairwise` suffixes.

All public harnesses also accept direct tracking configuration:

```python
from apnet_pt.training_tracking import WandbConfig

model.train(
    n_epochs=50,
    model_path="/absolute/path/model.pt",
    wandb_config=WandbConfig(
        mode="online",
        project="qcmlforge",
        group="apnet2-ablation",
        tags=("apnet2", "seed-42"),
    ),
)
```

Metrics are emitted once per evaluation boundary. Pre-training evaluation is
`epoch=0`; completed epochs are `1..n_epochs`. Primary namespaces are
`train/mae/*`, `val/mae/*`, `train/loss_sum`, and `val/loss_sum`. `loss_sum` is
not sample-normalized. NaN and infinite values are logged; the model harness,
not W&B, controls divergence handling.

Each successful run uploads a checkpoint only when the best validation model
changes and once at the end. The local `model_path` retains best-checkpoint
semantics. If the last epoch is also best, one artifact version receives
`best`, `final`, and `latest`.

A checkpoint passed back into `train()` is a warm start, not an exact resume. It
does not restore optimizer, scheduler, RNG, split, best loss, or epoch state;
the new call creates a new W&B run and records checkpoint lineage.

For DDP, only global rank 0 owns the W&B run and artifacts. Reduced metrics are
logged after synchronization. Historical distributed pairwise MAEs used a
component-count denominator and read low by that factor; do not compare them
directly with current runs. `train_ddp_slurm.py` avoids nested `mp.spawn`, but
multi-node SLURM tracking remains experimental until a two-node smoke test is
completed.
