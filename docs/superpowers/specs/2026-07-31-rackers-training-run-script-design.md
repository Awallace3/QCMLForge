# Rackers Training Run Script Design

## Goal

Add an executable root-level `run.sh` that trains both Rackers Thole damping
variants sequentially using the Splinter SAPT0/aug-cc-pVDZ settings from the
`AM-DimerParam` example in `train_ap3d3_saptdft_local_1.sh`.

## Execution Model

The script will use `set -euo pipefail`. It will invoke `train_models.py` first
with `RackersTholeDampingModel` and then with
`RackersTholeDampingOverlapModel`. The second run will start only if the first
run exits successfully.

A shared Bash argument array will hold all common training options. This avoids
duplicating configuration while keeping both model invocations explicit and
readable.

## Default Configuration

The script will reproduce the relevant Splinter example settings:

- iteration/random seed: `1`
- model directory: `./models/ap3_saptpbe0/1`
- pretrained atom model: `./models/ap3_saptpbe0/1/am_ap2_1.pt`
- pretrained HFVR/VW atom-type model:
  `./models/ap3_saptpbe0/1/atp_hfvr_1.pt`
- dataset root: `../qcmlforge/data_dir`
- pairwise dataset specification: `2`
- epochs: `25`
- learning rate: `5e-5`
- architecture: `n_rbf=8`, `n_neuron=64`, `n_embed=8`
- in-memory dataset: `True`
- DDP world size: fixed at `1`; any other `WORLD_SIZE_DDP` value is rejected
  before directory creation or training
- OMP threads: `16`, forwarded to each Rackers harness training call

The script will not pass `n_params`, `dimer_eval_type`, `param_start_mean`, or
`param_start_std`. The fixed Rackers harnesses select their own dimer modes and
use their route-specific four-head initialization defaults.

## Environment Overrides

Every path and training setting above except DDP world size will be
configurable through an environment variable while retaining the listed local
default. `WORLD_SIZE_DDP` remains environment-visible but accepts only `1`,
because the Rackers harnesses support single-process training only.
`OMP_NUM_THREADS` is effective at the final Rackers training call. Its launcher
default is `16`; a bare pairwise CLI and a direct `train_pairwise_model` call
default to `8`, while a bare atom CLI retains its default of `1`. Explicit CLI
values are unchanged. The Python executable will also be configurable,
defaulting to `python3`.

The default output checkpoints will be distinct:

- `${MODEL_DIR}/rackers_thole_${ITER}.pt`
- `${MODEL_DIR}/rackers_thole_overlap_${ITER}.pt`

Callers may override each output independently. Existing output checkpoints
retain `train_models.py`'s current resume behavior.

## Error Handling

The script will validate `WORLD_SIZE_DDP` as exactly `1`, then create
`MODEL_DIR` before training. Shell strict mode ensures that undefined variables,
command failures, and pipeline failures stop the script. No checkpoint deletion
or automatic overwrite logic will be added.

## Testing

A focused shell-script test will substitute a fake Python executable, run the
script without starting model training, and assert:

1. exactly two training invocations occur;
2. the pure model runs before the overlap model;
3. both receive the shared Splinter configuration and pretrained paths;
4. each receives its distinct output path; and
5. environment overrides are honored; and
6. launcher/default/explicit OMP values reach the final route-specific training
   call while Rackers world size remains `1`.

The implementation will also be checked with `bash -n run.sh`.
