# MACE–AP3D3 production benchmark

`scripts/slurm/benchmark.py` generates and optionally submits a locked,
multi-seed comparison of plain AP3D3 and the four canonical MACE routes.
Generation is safe by default: **no job is submitted unless `--submit` is
present**.

This workflow is distinct from the one-epoch verification scripts documented
in `mace-apnet-slurm.md`. It is intended for production model-quality runs,
but its output is scientific evidence only after the dataset provenance,
leakage audit, CUDA gate, and cluster controls in
`mace-apnet-validation-spec.md` have been approved. The external PolarMACE
artifact remains subject to its ASL restrictions; generated feature caches are
controlled internal artifacts and must not be published as redistributable
model content.

## Comparison matrix

All routes use one immutable training and physics configuration:

| Route | Description | Dependency |
|---|---|---|
| `BASE` | Plain `APNet3-fused-d3`, no MACE features | Prepared AP3 datasets only |
| `H1` | Legacy AP3 properties plus final-layer MACE scalars | MACE feature cache |
| `H2` | Legacy AP3 properties plus all MACE scalars/norms | MACE feature cache |
| `DirectPolar` | PolarMACE direct q/μ plus learned completion heads | Matching atomic-head seed |
| `AtomHead` | Learned q/μ/Q heads on frozen MACE features | Matching atomic-head seed |

The initial pair corpus is locked to AP3 `spec_type=2`:

- `1600K_train_dimers-fixed.pkl`
- `1600K_test_dimers-fixed.pkl`

Atomic q/μ/Q supervision combines the PBE0/MBIS monomer sources:

- `monomers_ap3_spec_1_pbe0.pkl`
- `monomers_ap3_spec_5_pbe0.pkl`

Atomic reports compare only MBIS charge, dipole, and Cartesian
symmetric-traceless quadrupole predictions. Response properties not present in
those files are not reported as independently supervised atomic science.

## Frozen input contract

The JSON configuration must satisfy
`docs/schemas/mace-benchmark-v1.schema.json`; split and pair-result records are
documented by `mace-benchmark-split-v1.schema.json`,
`mace-benchmark-provenance-v1.schema.json`, and
`mace-benchmark-result-v1.schema.json`. Every input has an explicit path
and SHA-256. The split manifest must contain:

- an approved provenance manifest tying pair/atomic methods, columns, units,
  licenses, and file digests to the configured target names;
- explicit processed-dataset index bases and train/validation/test membership;
- one leakage-group ID per pair row;
- no group overlap across pair partitions;
- an audit status of `passed`, grouping-key description, and auditor identity;
- zero atomic-training/pair-test monomer overlap under the canonical
  cross-dataset policy;
- explicit atomic membership and molecule group IDs for both PBE0/MBIS files.

The controller validates every digest and writes
`benchmark.lock.json`. Existing locks are immutable; changing any field
requires a new `benchmark_id`.

The physics file is a serialized `PhysicsConfig` plus its `physics_hash`.
Non-default Thole/SCF overrides are rejected until the production factory can
propagate those values without silently dropping them. Matched comparisons are
currently locked to damped-CLIFF; AMOEBA and undamped electrostatics are not
accepted because BASE would not share the same classical contract.

## Generate jobs

```bash
python scripts/slurm/benchmark.py --config /path/to/benchmark.json
```

Generated files are placed under:

```text
<output_root>/<benchmark_id>/
  benchmark.lock.json
  submission/
    dataset.sbatch
    prepare.sbatch
    baseline.sbatch
    hybrid.sbatch
    atomic.sbatch
    polar.sbatch
    report.sbatch
    *.tasks.json
    jobs.json
  data/
  feature-cache/
  atomic/
  runs/
  results/
  plots/
  slurm/
```

The dependency graph keeps the MACE-free baseline independent of MACE feature
preparation:

```text
dataset ─┬─> BASE array
         └─> MACE feature preparation ─┬─> H1/H2 array
                                      └─> atomic-head array
                                            └─> DirectPolar/AtomHead array

BASE, H1/H2, and DirectPolar/AtomHead ──afterany──> report/plot job
```

Dataset preparation and MACE feature preparation are different jobs. Dataset
preparation produces the shared AP3/MBIS data representation. MACE preparation
runs the verified frozen PolarMACE backbone once per unique monomer and stores
both canonical feature modes.

## Submit

After inspecting the lock, task tables, and generated SBATCH directives:

```bash
python scripts/slurm/benchmark.py \
  --config /path/to/benchmark.json \
  --submit
```

The controller submits seven jobs, four of which are arrays. The final report
job uses `afterany` so failures still produce clearly partial plots. A submission receipt
with scheduler job IDs and exact commands is written to
`submission/submitted.json`. Reusing a benchmark with an existing receipt is
rejected to prevent duplicate submission.

Production workers require CUDA. This requirement does not claim that CUDA or
the scheduler has been validated on a particular cluster; those remain
external V2/V3 gates.

## Collect and plot partial results

Plotting can be run while arrays are still active:

```bash
python scripts/slurm/benchmark.py \
  --config /path/to/benchmark.json \
  --plot
```

Outputs include:

- `plots/accuracy.png`: total test MAE/RMSE by route with seed-level 95% CIs;
- `plots/components.png`: ELST/EXCH/IND/DISP test MAE with 95% CIs;
- `plots/learning_curves.png`: per-seed validation losses;
- `plots/baseline_delta.png`: paired-seed total-MAE changes versus BASE;
- `plots/atomic_properties.png`: PBE0/MBIS q/μ/Q test MAEs for both atomic routes;
- `plots/resources.png`: elapsed time and peak RSS;
- `plots/metrics.csv`: machine-readable pair-run metrics;
- `plots/atomic_metrics.csv`: machine-readable q/μ/Q metrics;
- `plots/aggregate.csv`: route means and seed-level 95% confidence intervals;
- `plots/baseline_deltas.csv`: paired per-seed BASE-relative total-MAE changes;
- `plots/summary.json`: missing, incomplete, and invalid result records;
- `plots/STATUS.txt`: explicit completion label.

A plot is marked `COMPLETE` only when every canonical route/seed has a valid
`PASS` result that exhausted its epoch budget or ended under the locked
early-stopping rule. Otherwise every figure is watermarked `PARTIAL`, and the
status reports both complete runs and merely reported result files. Failed tasks
publish `FAIL` records before returning a nonzero scheduler status; absent
external prerequisites such as CUDA are recorded separately as `BLOCKED`.

## Scientific interpretation

The benchmark reports per-component and total MAE/RMSE, learning curves, and
resources. BASE-relative improvement should be computed across matched seeds.
Do not claim MACE superiority from:

- incomplete matrices;
- smoke fixtures;
- training or validation error alone;
- a split without a passed leakage audit;
- unapproved target provenance;
- atomic q/μ/Q results as validation of unsupervised response properties;
- a single seed.
