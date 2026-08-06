# Rackers Training Run Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an executable `run.sh` that sequentially trains the pure and overlap Rackers Thole damping models with environment-overridable Splinter defaults, while validating the single-process Rackers constraint.

**Architecture:** A root-level Bash script defines validated defaults and one shared argument array, rejects `WORLD_SIZE_DDP` values other than `1`, then performs two explicit `train_models.py` invocations with distinct model identifiers and output checkpoints. The CLI forwards `OMP_NUM_THREADS` through `train_pairwise_model` to each Rackers harness training call. A pytest test replaces Python with a recorder executable so command construction and ordering are verified without training.

**Tech Stack:** Bash, Python 3, pytest, `train_models.py` CLI.

## Global Constraints

- Use `set -euo pipefail`; the overlap run starts only after the pure run succeeds.
- Default configuration must match the Splinter `AM-DimerParam` example in `train_ap3d3_saptdft_local_1.sh`.
- Do not pass `n_params`, `dimer_eval_type`, `param_start_mean`, or `param_start_std`.
- Default outputs are `${MODEL_DIR}/rackers_thole_${ITER}.pt` and `${MODEL_DIR}/rackers_thole_overlap_${ITER}.pt`.
- Every path and training setting except Rackers world size is environment-overridable.
- `WORLD_SIZE_DDP` remains environment-visible but must be exactly `1` and is validated before directory creation or training.
- `OMP_NUM_THREADS` must reach `apnet.train` as `omp_num_threads_per_process`; its launcher default is `16`, omitted pairwise CLI and direct `train_pairwise_model` calls default to `8`, omitted atom CLI calls default to `1`, and explicit values are preserved.

---

### Task 1: Sequential Rackers training launcher

**Files:**
- Create: `run.sh`
- Create: `tests/test_run_script.py`

**Interfaces:**
- Consumes: `train_models.py` CLI routes `RackersTholeDampingModel` and `RackersTholeDampingOverlapModel`.
- Produces: executable `run.sh`; environment variables `PYTHON`, `ITER`, `MODEL_DIR`, `AM_MODEL_PATH`, `ATOM_TYPE_PARAM_MODEL_PATH`, `DATA_DIR`, `RANDOM_SEED`, `N_EPOCHS`, `LEARNING_RATE`, `N_RBF`, `N_NEURON`, `N_EMBED`, `SPEC_TYPE_AP`, `DS_IN_MEMORY`, `WORLD_SIZE_DDP`, `OMP_NUM_THREADS`, `RACKERS_MODEL_OUT`, and `RACKERS_OVERLAP_MODEL_OUT`.

- [ ] **Step 1: Write the failing command-construction test**

Create `tests/test_run_script.py`. The test must create an executable temporary Python recorder whose shebang launches Python and whose body appends `json.dumps(sys.argv[1:])` to `CALL_LOG`. Run `bash run.sh` from the repository root with all public environment variables set to unmistakable temporary/test values.

Assert that exactly two JSON argument arrays were recorded; the first contains `--train_apnet RackersTholeDampingModel`; the second contains `--train_apnet RackersTholeDampingOverlapModel`; both contain identical overridden shared options; and their `--ap_model_path` values are the independently overridden pure and overlap outputs. Assert neither command contains `--n_params`, `--dimer_eval_type`, `--param_start_mean`, or `--param_start_std`.

Also add a default-value test that runs through the same recorder with only `PYTHON` and `CALL_LOG` overridden and asserts the documented default atom model, HFVR/VW model, dataset, spec, architecture, optimization settings, and output names. Add boundary tests proving OMP reaches the final Rackers `train` call and invalid world sizes fail before directory creation or invocation.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_run_script.py -v
```

Expected: FAIL because root-level `run.sh` does not exist.

- [ ] **Step 3: Implement the minimal script**

Create `run.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ITER="${ITER:-1}"
MODEL_DIR="${MODEL_DIR:-./models/ap3_saptpbe0/${ITER}}"
AM_MODEL_PATH="${AM_MODEL_PATH:-${MODEL_DIR}/am_ap2_${ITER}.pt}"
ATOM_TYPE_PARAM_MODEL_PATH="${ATOM_TYPE_PARAM_MODEL_PATH:-${MODEL_DIR}/atp_hfvr_${ITER}.pt}"
DATA_DIR="${DATA_DIR:-../qcmlforge/data_dir}"
RANDOM_SEED="${RANDOM_SEED:-${ITER}}"
N_EPOCHS="${N_EPOCHS:-25}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
N_RBF="${N_RBF:-8}"
N_NEURON="${N_NEURON:-64}"
N_EMBED="${N_EMBED:-8}"
SPEC_TYPE_AP="${SPEC_TYPE_AP:-2}"
DS_IN_MEMORY="${DS_IN_MEMORY:-True}"
WORLD_SIZE_DDP="${WORLD_SIZE_DDP:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
RACKERS_MODEL_OUT="${RACKERS_MODEL_OUT:-${MODEL_DIR}/rackers_thole_${ITER}.pt}"
RACKERS_OVERLAP_MODEL_OUT="${RACKERS_OVERLAP_MODEL_OUT:-${MODEL_DIR}/rackers_thole_overlap_${ITER}.pt}"

case "${WORLD_SIZE_DDP}" in
    1)
        ;;
    *)
        printf 'Error: WORLD_SIZE_DDP must be exactly 1 (got %q)\n' \
            "${WORLD_SIZE_DDP}" >&2
        exit 2
        ;;
esac

case "${DS_IN_MEMORY}" in
    [Tt][Rr][Uu][Ee])
        DS_IN_MEMORY=true
        ;;
    [Ff][Aa][Ll][Ss][Ee])
        DS_IN_MEMORY=false
        ;;
    *)
        printf 'Error: DS_IN_MEMORY must be true or false (got %q)\n' \
            "${DS_IN_MEMORY}" >&2
        exit 2
        ;;
esac

mkdir -p "${MODEL_DIR}"

COMMON_ARGS=(
    --am_model_path "${AM_MODEL_PATH}"
    --atom_type_param_model_path "${ATOM_TYPE_PARAM_MODEL_PATH}"
    --random_seed "${RANDOM_SEED}"
    --n_epochs "${N_EPOCHS}"
    --n_rbf "${N_RBF}"
    --n_neuron "${N_NEURON}"
    --n_embed "${N_EMBED}"
    --data_dir "${DATA_DIR}"
    --spec_type_ap "${SPEC_TYPE_AP}"
    --lr "${LEARNING_RATE}"
)

if [[ "${DS_IN_MEMORY}" == true ]]; then
    COMMON_ARGS+=(--ds_in_memory True)
fi

COMMON_ARGS+=(
    --world_size_ddp "${WORLD_SIZE_DDP}"
    --omp_num_threads "${OMP_NUM_THREADS}"
)

"${PYTHON}" -u ./train_models.py \
    --train_apnet RackersTholeDampingModel \
    --ap_model_path "${RACKERS_MODEL_OUT}" \
    "${COMMON_ARGS[@]}"

"${PYTHON}" -u ./train_models.py \
    --train_apnet RackersTholeDampingOverlapModel \
    --ap_model_path "${RACKERS_OVERLAP_MODEL_OUT}" \
    "${COMMON_ARGS[@]}"
```

Mark it executable with `chmod +x run.sh`.

- [ ] **Step 4: Run focused tests and syntax verification**

Run:

```bash
python -m pytest tests/test_run_script.py -v
bash -n run.sh
```

Expected: all tests PASS and Bash syntax validation exits 0.

- [ ] **Step 5: Run relevant CLI regression and inspect the script**

Run:

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "cli or dispatch" -q
python train_models.py --help | grep -E "RackersTholeDamping(Model|OverlapModel)"
git diff --check
```

Expected: selected tests PASS, help names both model routes, and the diff check emits no output.

- [ ] **Step 6: Commit**

```bash
git add run.sh tests/test_run_script.py
git commit -m "feat(training): add sequential Rackers launcher"
```
