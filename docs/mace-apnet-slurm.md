# MACE–AP3D3 small SLURM verification workflow

These scripts implement only a reproducible **one-epoch smoke verification
gate**. They are not a production training path: multi-epoch/full-dataset
lifecycles and full-scale submission are not implemented. MACE execution is
currently eager and requires `--skip_compile`; CUDA and real `sbatch` parity
remain external gates.

## Installation and local gate

Use the pinned optional environment and verify the checked-in wiring fixtures:

```bash
python -m pip install -e '.[mace]'
python scripts/make_mace_ap3d3_smoke_data.py --check
python -m pytest tests/test_mace_one_epoch.py tests/test_mace_slurm_scripts.py -q
```

The PolarMACE foundation checkpoint is an external Academic Software License
(ASL) artifact for approved academic, non-commercial internal use. Never commit,
embed, redistribute, or deserialize it before checking its trusted SHA-256.
Compute jobs use only a local artifact with `--mace_offline`; the scripts contain
no download path.

## Create the immutable physics record

The pair job requires a JSON physics record and its file digest. For the current
matched small gate:

```bash
python - <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
from apnet_pt.mace.schema import PhysicsConfig

config = PhysicsConfig()
record = asdict(config)
record["physics_hash"] = config.physics_hash
Path("agent_scratch/physics-default.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY
sha256sum agent_scratch/physics-default.json
```

## Dry-run the complete dependency matrix

Set paths and digests explicitly; do not use `AUTO`, URLs, or model aliases:

```bash
export RUN_ROOT="$PWD/agent_scratch/slurm-runs"
export MATRIX_ID="small-$(git rev-parse --short HEAD)"
export MACE_MODEL_PATH=/approved/path/MACE-POLAR-1-S.model
export MACE_MODEL_SHA256=e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510
export PAIR_DATA_PATH="$PWD/tests/dataset_data/mace_ap3d3_smoke.pkl"
export ATOM_DATA_PATH="$PWD/tests/dataset_data/mace_atomic_properties_smoke.pkl"
export AM_MODEL_PATH="$PWD/tests/test_models/ap3_ensemble_0/am_3.pt"
export ATOM_TYPE_PARAM_MODEL_PATH="$PWD/tests/test_models/ap3_ensemble_0/am_h+1_3.pt"
export ATOM_TYPE_PARAM_MODEL_PATH2="$PWD/tests/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
export AM_MODEL_SHA256="$(sha256sum "$AM_MODEL_PATH" | awk '{print $1}')"
export ATOM_TYPE_PARAM_MODEL_SHA256="$(sha256sum "$ATOM_TYPE_PARAM_MODEL_PATH" | awk '{print $1}')"
export ATOM_TYPE_PARAM_MODEL_SHA256_2="$(sha256sum "$ATOM_TYPE_PARAM_MODEL_PATH2" | awk '{print $1}')"
export PHYSICS_CONFIG_PATH="$PWD/agent_scratch/physics-default.json"
export PHYSICS_CONFIG_SHA256="$(sha256sum "$PHYSICS_CONFIG_PATH" | awk '{print $1}')"
export ELECTROSTATICS_MODE=damped-cliff
export SMALL_VERIFICATION_APPROVED=1
DRY_RUN=1 bash scripts/slurm/submit_mace_ap3d3_matrix.sh
```

After reviewing all 19 commands and unique run IDs, remove `DRY_RUN=1` to submit
one preparation job, three seed-specific atomic-head jobs, and 15 pair jobs.
Dependencies use `afterok`: each atomic job waits for feature preparation;
DirectPolar and AtomHead wait for the corresponding seed's atomic job; BASE,
H1, and H2 wait for preparation. A failed prerequisite cannot release a pair
job.

## Run and manifest layout

Each `RUN_ID` owns:

```text
logs/  cache/  data/  diagnostics/  tmp/  checkpoints/  manifest.json
```

Preparation atomically writes cache entries, skips existing entries only after
identity/tensor validation, and writes `cache/features/COMPLETE.json` last.
Consumers reject a cache without that completion manifest. Manifests record the
source commit, environment, dataset/split/preprocessing/physics hashes, feature
schemas, every submodel path and digest, seed, parameter counts, elapsed time,
and maximum resident memory.

Existing output checkpoints are always rejected. There is no implicit resume or
overwrite. Feature preparation is entry-level restartable. Current atomic and
pair harnesses only write final one-epoch checkpoints and do not expose a
periodic SIGTERM checkpoint hook; their manifests therefore record
`requeue_safe_checkpoint: false`. Requeueing interrupted training must use a new
run ID unless no output checkpoint was created.

## Remaining cluster preflight

The scripts initially request exactly one GPU and one task and invoke exactly one
`srun`. Until CUDA parity is approved, preparation and training retain the
validated CPU execution policy even on that reserved verification node. Before
any larger stage, independently verify the CUDA driver/wheel,
GPU memory, CPU/GPU prediction parity, cache online parity, and the chosen
internal-spawn versus `torchrun` policy. No CUDA or multi-node validation is
claimed by the checked-in workflow.
