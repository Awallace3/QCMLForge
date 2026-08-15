# MACE–AP3D3 validation specification

**Status:** Scoping draft v0.1 — not yet an approved scientific protocol
**Implementation under test:** `6c9118bc2c8ef2a1517075858d6aa19730f3b03a`
**Related contracts:** [implementation spec](mace-apnet-implementation-spec.md), [environment policy](mace-apnet-environment.md), [SLURM smoke workflow](mace-apnet-slurm.md)

## 1. Purpose and non-claims

This document defines the validation ladder required after implementation of:

- `MACE-AP3D3-DirectPolar`;
- `MACE-AP3D3-H1`;
- `MACE-AP3D3-H2`;
- `MACE-AP3D3-AtomHead`;
- the matched `APNet3-fused-d3` baseline.

Checked-in atomic targets and one-epoch smoke runs are wiring evidence only. They
must not be used to claim atomic-property accuracy, SAPT accuracy, CUDA parity,
cluster readiness, or production readiness.

The initial scope decisions are:

- validate the **full ladder**, not systems or science in isolation;
- anchor pair-model comparisons on the **existing AP3 production data**, after
  its target-method and provenance audit;
- qualify the **current production cluster**, with an exact hardware profile to
  be filled in before CUDA execution;
- require both **BASE-relative** and **absolute scientific-quality** gates.

## 2. Validation principles

1. Use the same immutable `PhysicsConfig`, data split IDs, labels, and evaluation
   code for all five controlled experiments.
2. Freeze data provenance, split hashes, model/backbone digests, feature schemas,
   dependency versions, and metric definitions before full training.
3. Select models on validation data only. Evaluate the locked test set once per
   approved comparison; do not tune against it.
4. Group related geometries and repeated monomer pairs in one split to prevent
   geometry and chemical-identity leakage.
5. Report each seed separately and aggregate with confidence intervals. Do not
   report only the best seed.
6. Separate implementation correctness, systems parity, and scientific quality.
   Passing an earlier gate is necessary but not evidence for a later gate.
7. Preserve failed-run manifests and diagnostics. Never silently omit NaN,
   non-converged induction, unsupported-element, or out-of-memory cases.

## 3. Required evidence bundle

Every non-unit validation run must write an immutable run directory containing:

```text
manifest.json                 # commit, route, seed, hashes, hardware, software
split.json                    # immutable train/validation/test IDs and group keys
metrics.json                  # aggregate metrics and acceptance decisions
predictions.parquet           # IDs, references, predictions, components, diagnostics
checkpoint/                   # QCMLForge checkpoint only; external backbone excluded
logs/                         # stdout/stderr and scheduler metadata
diagnostics/                  # induction, invalid cases, memory, timing, parity details
environment/                  # pip/conda lock, torch/CUDA/driver/GPU information
```

The manifest must include the PolarMACE artifact SHA-256, all route-specific
submodel digests, dataset and split hashes, feature schema, physics hash,
checkpoint schema version, source commit, dirty-tree state, command line, and
license acknowledgment. A dirty source tree is a failed controlled run.

## 4. Gated validation ladder

### V0 — Protocol and data freeze

| Check | Requirement | Pass criterion |
|---|---|---|
| V0.1 | Select the SAPT target level and D3 preset | One documented target convention; no mixed unconverted labels |
| V0.2 | Audit and freeze the existing AP3 production corpus | Exact `spec_type`/files, source, license, units, methods, component definitions, counts, and digests recorded; no silent target mixing |
| V0.3 | Approve atomic supervision for DirectPolar completion and AtomHead | Real q/μ/Q/response labels with units, origin convention, quadrupole convention, and provenance |
| V0.4 | Freeze leakage-safe splits | Grouped split IDs and hash; no monomer-pair/group overlap across partitions |
| V0.5 | Preregister metrics and thresholds | Thresholds approved before production training or test-set evaluation |
| V0.6 | Freeze controlled hyperparameters | Route-specific exceptions documented; common physics and training budget fixed |

**Stop:** A/C scientific training is blocked until V0.3 is satisfied. Full model
comparison is blocked until V0.1–V0.6 are approved.

### V1 — Local correctness and CPU reproducibility

Current evidence includes the ordinary test suite, real local PolarMACE smoke,
fixture determinism, checkpoint-v3 reconstruction, and shell/static checks. The
following evidence must be captured in CI rather than retained only in an agent
session:

| Check | Requirement | Pass criterion |
|---|---|---|
| V1.0 | Clean built-wheel qualification | Install a built wheel in a clean environment; base import works without MACE; optional install includes vendored `qcml_dftd3` and required package data such as `reference-c6.pt` |
| V1.1 | Base import without optional MACE dependencies | Import succeeds in base environment |
| V1.2 | Pinned real-backbone CPU integration | All `mace_integration` tests pass with the verified local artifact; missing required artifact is reported as `BLOCKED`, never `PASS` |
| V1.3 | Repeatability | Two clean CPU runs produce identical split/config hashes and predictions within the registered dtype tolerance |
| V1.4 | Checkpoint portability | Fresh process reload reproduces predictions and verifies every external digest |
| V1.5 | Negative checkpoint/cache cases | Wrong route, digest, schema, split, physics hash, incomplete cache, and offline miss fail before training |
| V1.6 | Independent physics oracles | Raw external/reference inputs, outputs, generator version/command, parameters, units, and tolerances validate CLIFF, undamped, induction, and D3; close-contact and non-convergent cases included |
| V1.7 | Prepared-cache producer/consumer path | The real preparation script feeds the real consumer with cache hits and no backbone calls; top-level/nested dataset, preprocessing, and split identities agree exactly |

AMOEBA is quarantined from scientific or release acceptance until an agreed
AMOEBA/HIPPO convention matches an independent oracle within a preregistered
tolerance. Its current explicit implementation/rejection tests are correctness
evidence only.

### V2 — Single-GPU CUDA parity and resource preflight

Run each of BASE, H1, H2, DirectPolar, and AtomHead on the target cluster GPU and
software image. Before execution, freeze this cluster profile:

| Field | Required value |
|---|---|
| cluster/partition/account/QoS | TBD from current production cluster |
| GPU model(s), count, and memory | TBD; first gate uses one dedicated GPU |
| driver, CUDA runtime, and PyTorch wheel | TBD and captured verbatim |
| CPU model/RAM/local scratch | TBD |
| container/module/lock digest | TBD |

Cross-architecture exactness is not required; paired comparisons use the same
physical GPU. A second GPU architecture is a release-candidate portability gate.

| Check | Requirement | Pass criterion |
|---|---|---|
| V2.1 | Environment compatibility | Pinned Python/Torch/MACE/e3nn stack loads with the installed driver and CUDA runtime |
| V2.2 | CPU/GPU inference parity | Compare features, atomic properties, classical/residual ledgers, components, and totals; proposed float32 `atol=rtol=2e-4`, float64 `1e-7`; no case-specific exclusions |
| V2.3 | Online/cache parity | Online and prepared-cache GPU predictions meet the same tolerance and identity checks |
| V2.4 | Invariance on GPU | Rotation, translation, permutation, A/B swap, and batch-order tests pass at GPU tolerance |
| V2.5 | Gradient sanity | Expected trainable modules receive finite nonzero gradients; backbone and inactive modules remain gradient-free; proposed one-step float32 gradient/update `atol=rtol=5e-4` |
| V2.6 | Repeated-run determinism | Deterministic mode reruns satisfy the approved same-device tolerance and record nondeterministic operators |
| V2.7 | Resource envelope | Peak host/GPU memory, samples/s, cache size, and elapsed time recorded for every route |
| V2.8 | Failure behavior | OOM, corrupt cache, unsupported element, and induction non-convergence terminate or apply the configured explicit policy |

Mixed precision and compilation remain separate experiments. Eager float32 must
pass first. No AMP or compiled result may replace the eager reference until its
own parity gate is approved.

### V3 — Real SLURM dependency workflow

Use the checked-in one-epoch scripts without changing their verification-only
scope.

| Check | Requirement | Pass criterion |
|---|---|---|
| V3.1 | Preparation job | Completion manifest written last; counts/digests complete; restart reuses only validated entries |
| V3.2 | Atomic jobs | Three seeds each for DirectPolar completion and AtomHead; finite loss, checkpoint, reload, and manifests |
| V3.3 | Pair jobs | BASE plus four routes × three seeds complete one epoch and inference |
| V3.4 | Dependency enforcement | Failed preparation/atomic job prevents dependent jobs from starting |
| V3.5 | Offline execution | Compute-node logs show no network access or implicit artifact download |
| V3.6 | Scheduler evidence | Job IDs, states, exit codes, `sacct` resources, logs, and run manifests archived |

This gate validates scheduler wiring only, not model quality.

### V4 — Atomic-property scientific validation

Use independently sourced, non-wiring atomic labels. Evaluate both
DirectPolar completion and AtomHead; also record raw PolarMACE q/μ behavior where
its contract supplies those quantities.

Required metrics:

- charge conservation error and atomic charge MAE/RMSE;
- molecular dipole reconstruction and vector MAE/angular error;
- Cartesian symmetric-traceless quadrupole error using a fixed origin;
- polarizability/response MAE, relative error, positivity, and physical bounds;
- covariance/invariance residuals in float64 and float32;
- performance by element, charge, multiplicity, molecule size, and chemistry;
- calibration and failure counts, not only aggregate averages.

Required negative controls:

- legacy AP3 atomic models;
- direct/raw PolarMACE outputs where defined;
- element-mean or similarly trivial predictor;
- shuffled-label training to detect leakage.

A/C may proceed to scientific pair comparison only after their atomic models meet
approved per-property thresholds and show no catastrophic subgroup failures.
Provisional thresholds for domain-owner review are: exact physical constraints
(charge residual and Q symmetry/trace at most `1e-6` in canonical units), and an
upper 95% confidence bound on degradation versus the legacy provider below the
larger of 5% of legacy MAE or: q `0.01 e`, atomic μ norm `0.02 e·bohr`, Q
Frobenius norm `0.05 e·bohr²`, HFVR `0.02`, and width `0.02 bohr`. These are
proposals, not approved targets. Absolute domain-quality limits must also be
set from the intended AP3 use case rather than inferred from BASE.

### V5 — SAPT component benchmark

Train the five controlled experiments with identical splits, physics settings,
optimizer policy, stopping rule, and compute budget unless a difference was
preregistered.

Report for ELST, EXCH, IND, DISP, and total interaction energy:

- MAE, RMSE, median absolute error, signed bias, and high-error quantiles;
- bootstrap confidence intervals and all seed values;
- learning curves and validation-selected epoch;
- results by interaction class, chemical subgroup, distance bin, system size,
  charge/multiplicity, and in-domain versus out-of-domain split;
- induction convergence and invalid-case rates;
- parameter count, training time, inference throughput, and peak memory.

Mandatory baselines/ablations:

1. matched `APNet3-fused-d3` BASE;
2. classical-only ELST + induction + D3 with zero neural residual;
3. H1 versus H2 under identical capacity/budget;
4. each MACE route with MACE features removed or replaced by a controlled null;
5. `no_disp_nn` where scientifically relevant;
6. seed and data-volume sensitivity.

**Provisional model-quality rule:** for total and every component, require the
upper 95% confidence bound on `MAE_route - MAE_BASE` to be below
`max(0.02 kcal/mol, 0.05 × MAE_BASE)`. Require total-energy Q95 not to worsen by
more than 10%, and no preregistered stratum with at least 100 cases to worsen by
more than 20% without an explicit scope limitation. A superiority claim requires
the upper bound on the paired total-MAE difference to be below zero. These
BASE-relative rules must be combined with absolute per-component and total
limits approved before test evaluation.

### V6 — Robustness and extrapolation

| Check | Required challenge set |
|---|---|
| V6.1 | Separation scans spanning repulsive contact, equilibrium, `r_cut_im`, and asymptotic range |
| V6.2 | Held-out monomer identities and interaction classes |
| V6.3 | Larger systems than the training distribution |
| V6.4 | Supported ionic/radical cases after charge/spin validation |
| V6.5 | Atom-order, fragment-order, rigid-motion, and batching metamorphic tests |
| V6.6 | Near-singular/close-contact and induction non-convergence stress cases |
| V6.7 | Unsupported elements and malformed metadata fail explicitly |

Energy and first-derivative continuity must be checked at the neural cutoff.
Forces are not a supported scientific claim: the current frozen featurizer runs
the backbone under `torch.no_grad()` and detaches positions/features, while
prepared caches necessarily remove coordinate dependence. Therefore full
`-dE/dR` omits MACE-feature/property response. Force support requires a separate
architecture, cache, memory, and reference-validation contract.

### V7 — Scale, restart, and release gate

Before full production training:

- complete 32–256 and 1k–10k staged runs with resource scaling curves;
- set memory/time limits from measured envelopes rather than guesses;
- implement and validate periodic/requeue-safe checkpoints or explicitly retain
  the current new-run-only interruption policy;
- validate single-GPU versus the chosen multi-GPU strategy before multi-GPU use;
- reproduce one locked run from only its evidence bundle and approved external
  artifacts;
- conduct an independent review of metrics, exclusions, manifests, and license
  constraints.

CI tiers mirror the release ladder:

- **CI-0, every PR:** offline fixture/schema/CLI/cache/checkpoint/shell tests;
- **CI-1, merge queue:** full stub MACE suite, legacy regressions, deterministic
  reruns, and dtype seam tests;
- **CI-2, protected CPU runner (currently DISABLED/BLOCKED):** future mandatory
  real PolarMACE integration from an administrator-approved artifact after all
  external controls and policy attestation are established;
- **CI-3, protected GPU runner:** CPU/CUDA parity, deterministic rerun, memory,
  and controlled OOM cleanup;
- **CI-4, manual/weekly cluster:** real SLURM success/failure dependency matrix,
  TERM/OOM exercises, and 256/1k resource envelopes;
- **CI-5, release candidate:** second GPU architecture, clean reconstruction,
  license attestation, checkpoint portability, and performance regression.

Every executed pytest tier publishes JUnit and machine-readable check status.
A preflight-blocked tier publishes the terminal machine-readable report without
claiming that pytest or JUnit executed. `SKIP`, `BLOCKED`, and `PASS` are distinct.

### Local executable foundation

The local authority runs only CI-0, CI-1, and CI-2:

```bash
python scripts/validate_mace_local.py --tier CI-0 \
  --output-dir artifacts/mace-local
python scripts/validate_mace_local.py --tier CI-1 \
  --output-dir artifacts/mace-cpu-stub
QCMLFORGE_POLARMACE_ARTIFACT=/approved/read-only/MACE-POLAR-1-S.model \
python scripts/validate_mace_local.py --tier CI-2 \
  --mace-artifact "$QCMLFORGE_POLARMACE_ARTIFACT" \
  --output-dir artifacts/mace-cpu
python scripts/validate_mace_local.py --list-checks
```

CI-3, CI-4, and CI-5 are nonlocal and the command rejects them. Reports use
[`mace-local-validation-v1.schema.json`](schemas/mace-local-validation-v1.schema.json),
are written atomically as `validation-report.json`, and integrity-link each
existing evidence file with a relative path, byte count, and SHA-256. The
Python semantic validator is authoritative at generation time; equivalent
Draft 2020-12 conditionals are published for external validators without adding
a base runtime dependency. SIGTERM/SIGINT outside or during pytest produces a
terminal report or atomic `validation-interrupted.json` with no readiness claim.
Generated `artifacts/` and `junit-*.xml` are ignored (and CI-2 uses runner temp)
so one tier's evidence cannot dirty the next controlled source check. Stable
pair/atomic tests execute default-factory scope forwarding; the optional-stack
real factory plus zero-backbone prepared-cache path remains a CI-2 integration
check, while artifact-free tests exercise the real featurizer/cache seam directly.

| Status | Meaning | Aggregate exit |
|---|---|---|
| `PASS` | The named check ran and its criterion was observed | 0 only when every required selected check passes |
| `FAIL` | An executed check, artifact integrity check, or required JUnit criterion failed | 1 (takes precedence) |
| `BLOCKED` | A required external prerequisite prevented execution | 2 when no check failed |
| `SKIP` | Explicitly not applicable or not selected; never satisfies a required gate | nonzero if required |

CI-2 resolves the administrator-provided artifact only from
`QCMLFORGE_POLARMACE_ARTIFACT` (or the identical explicit CLI argument). It
requires the canonical 33,375,439-byte size and SHA-256
`e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510`
before any deserialization. The local check verifies that no Unix write
permission bits are present; this does **not** prove mount immutability. Absence
or a missing dependency is `BLOCKED`; wrong size/digest, wrong installed
version/commit, and any empty, skipped, failed, or erroneous integration JUnit
are `FAIL`. The successful preflight remains a separate report check with the
resolved locator, canonical size/digest, exact permission-bit property, and
Python/Torch/MACE/e3nn/graph-longrange identity. No artifact bytes enter evidence.

CI-2 is **DISABLED/BLOCKED** by an unconditional job guard. Confirmed external
blockers are: (1) no `polarmace-protected` GitHub Environment, (2) unprotected
`main`, and (3) no isolated matching self-hosted runner. A reviewed future
change may remove the guard only after administrators verify Environment
restrictions/review, branch protection, runner isolation/cleanup, and a
read-only approved mount. The authority also requires and records
`QCMLFORGE_CI2_POLICY_ATTESTED=true`, intended to be supplied only by that
protected Environment; a local environment value or repository YAML is an
assertion, not proof of those controls. Missing/false attestation is `BLOCKED`
and cannot produce CI-2 `PASS`. Trusted-main checkout clauses remain defense in
depth, not the security boundary.

Built-wheel qualification is a separate evidence-producing job and is not
claimed by a CI-0 authority PASS. It is performed outside the checkout and
repository `conftest.py` source injection:

```bash
python -m build --wheel --outdir dist
python scripts/ci/probe_built_wheel.py --wheel dist/qcmlforge-*.whl
# After installing that wheel in a fresh environment, run from /tmp:
python /path/to/probe_built_wheel.py --wheel /path/to/wheel \
  --checkout /path/to/checkout --installed
```

The probe enforces MACE-free base imports, optional-only dependency metadata,
required MACE adapter modules, loadable vendored `qcml_dftd3` data, and a wheel
denylist for foundation/checkpoint artifacts, bytecode, and caches.

`scripts/run_with_metrics.py` qualifies only local process behavior. It writes
versioned atomic JSON with argv/PID/process group, UTC and monotonic timing,
exit/signal/timeout/launch-failure outcomes, bounded TERM/KILL handling, and
normalized host RSS samples when available. Its capability record explicitly
sets GPU memory, GPU utilization, throughput, and SLURM accounting to false;
these records are not V2.7 or V7 evidence.

Static shell parsing and fake-`sbatch` tests are **CI-0 SLURM script wiring
only**. They are not scheduler execution, dependency-state, or cluster evidence.

> A local `PASS` is evidence only for the named CPU wiring check. It is not
> evidence of scientific accuracy, CUDA parity, scheduler execution, cluster
> readiness, production readiness, or a later release state.

Release states are explicit:

```text
IMPLEMENTED -> CPU_VERIFIED -> CUDA_VERIFIED -> SLURM_VERIFIED
            -> SCIENTIFICALLY_VALIDATED -> SCALE_APPROVED
```

No state may be inferred from a later-looking filename or a successful
checkpoint load.

## 5. Remaining blockers and resolved local findings

Scoped v2 producer/consumer identity and protected missing-artifact handling are
resolved local-foundation findings, not current known mismatches. CI-2 itself
has three confirmed external blockers: no protected GitHub Environment,
unprotected `main`, and no isolated self-hosted runner. Other out-of-scope
non-claims remain:

1. Local host metrics do not establish GPU allocation, utilization, throughput,
   or a performance envelope and are not V2.7/V7 evidence.
2. Pair/atomic training intentionally rejects resume and manifests are not
   requeue-safe. Interrupted training must fail closed with a new run ID until a
   separate exact optimizer/scheduler/RNG resume contract is implemented.
3. AMOEBA has a known scientific-oracle disagreement and remains quarantined.
4. Full forces are architecturally unsupported by detached frozen/cached features.
5. Independent CLIFF, undamped, induction, and D3 oracle provenance is not yet
   sufficient for publication-quality validation.

## 6. Initial executable tranche

The approved local-foundation tranche is limited to strict scoped prepared-cache
identity and completeness, CI-0/1/2 local evidence mechanics, clean built-wheel
qualification, protected CPU artifact preflight, local process-wrapper behavior,
and executable workflow tier mapping. The prepared-cache v2 manifest contains
explicit `pair` and `atomic` identities and deterministic monomer membership;
consumers must select one dataset kind and supply all three hashes. Explicit v1
top-level or scoped identities remain
readable, but missing identity values never compare as matches.

CUDA execution, real scheduler execution, scientific/data-quality validation,
forces, resume, AMOEBA oracle work, production-data audit, and artifact
redistribution remain outside this tranche. Static/fake-`sbatch` tests establish
SLURM script wiring only. No later release state may be inferred from completion
of this local work.

## 7. Thresholds requiring approval

The final spec must replace these placeholders before scientific execution:

- CPU/GPU prediction and gradient parity tolerances by dtype;
- atomic q/μ/Q/response acceptance thresholds;
- BASE non-inferiority margins for each SAPT component and total;
- minimum effect size required to claim a MACE-route improvement;
- bootstrap confidence level and resampling unit;
- allowed invalid/non-converged fraction;
- subgroup minimum sample sizes;
- resource ceilings and timeout policy;
- supported chemical domain, elements, charges, multiplicities, and size range.

## 8. Stop rules

Stop and escalate if:

- target labels mix methods, units, origins, or quadrupole conventions;
- train/validation/test groups leak related geometries or monomer identities;
- any route receives different physics, split IDs, or test-time filtering;
- the external artifact digest/license differs from the approved record;
- CPU/GPU or online/cache parity fails beyond the preregistered tolerance;
- atomic supervision is synthetic or wiring-only;
- induction failures, unsupported species, NaN/Inf, or OOM cases are discarded;
- a threshold would be selected after examining locked test results;
- cluster execution requires an unreviewed change to the pinned scientific stack.

## 9. Open decisions

1. Which exact existing AP3 `spec_type`/files, target level, and legal access path are approved?
2. What independent atomic-property dataset will supervise and evaluate A/C?
3. What grouping key defines leakage-safe in-domain and out-of-domain splits?
4. What absolute per-property/component/total limits supplement the proposed BASE-relative rules?
5. What are the current cluster's GPU, CUDA image, partition/account/QoS, and resource limits?
6. Should real PolarMACE integration run in dedicated CI, scheduled CI, or cluster preflight only?
7. Is eager float32 the sole first-release execution mode, or must float64/AMP be supported?
8. Is periodic/requeue-safe training checkpointing required before scale-up?
9. Which multi-GPU policy, if any, will be validated: internal spawn or `torchrun`?
10. What publication/release claims and ASL review are intended?
11. Is AMOEBA intended as a future release feature or an explicitly experimental mode?
