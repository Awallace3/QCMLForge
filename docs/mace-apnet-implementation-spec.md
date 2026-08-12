# MACE × AP3D3 Implementation Specification

**Status:** Proposed implementation contract
**Target branch/worktree:** `mace-apnet`
**Architecture reference:** [`docs/mace-apnet.html`](./mace-apnet.html)
**Primary entry point:** `train_models.py`

## 1. Goal

Implement the three MACE/AP3D3 atomic-property architectures and four public model options described in `docs/mace-apnet.html`, verify that each option can construct, train for at least one epoch, predict finite SAPT0 components, save, and reload on a small checked-in dataset, and only then provide SLURM `sbatch` jobs for larger training runs.

The architecture atlas contains three atomic-property routes and two hybrid pair topologies. The implementation is complete only when these four canonical, case-sensitive CLI options work:

```text
--train_apnet MACE-AP3D3-DirectPolar
--train_apnet MACE-AP3D3-H1
--train_apnet MACE-AP3D3-H2
--train_apnet MACE-AP3D3-AtomHead
```

H1 and H2 share the hybrid atomic-property route but are separate public options so checkpoints, logs, SLURM arrays, and topology comparisons are auditable. All four options must use the same AP3D3 residual-energy contract and the same long-range physics implementation.

## 2. Non-negotiable scientific contract

Every architecture must assemble the four SAPT0 components as

```text
ELST = E_MTP-MTP + ΔE_NN,elst
EXCH =              E_NN,exch
IND  = E_classical_induction + ΔE_NN,ind
DISP = E_D3 + ΔE_NN,disp
```

`ΔE_NN,disp` may be disabled with the existing `--no_disp_nn` behavior, but D3 must remain present.

The default loss compares the fully assembled four-component prediction with the original full SAPT labels. Dataset processing must not mutate labels differently for live versus precomputed classical terms. Any residual-target optimization is a separately named, checkpointed mode and must be mathematically equivalent across all four model options.

The shared long-range provider must support:

- MTP-MTP electrostatics:
  - damped CLIFF;
  - damped AMOEBA;
  - undamped;
- `classical_induction` using self-consistent induced dipoles and the existing Thole machinery;
- D3 using `qcml_dftd3.d3.resolve_d3_damping_parameters`;
- one documented atom-property convention and one energy-unit convention across A, B, and C;
- pairwise and per-dimer breakdowns so tests can detect missing or double-counted terms.

The current `DimerProp.set_forward()` API does not expose every combined electrostatics/induction/D3 combination. The implementation must not emulate an undamped or AMOEBA combined mode by silently selecting the existing CLIFF combined mode. Add a unified provider that calls the existing low-level MTP-MTP, induction, and D3 functions explicitly.

The physics configuration is immutable and hashed:

```python
@dataclass(frozen=True)
class PhysicsConfig:
    electrostatics_mode: Literal["damped-cliff", "damped-amoeba", "undamped"]
    electrostatics_parameters: tuple[float, ...]
    full_pair_edge_semantics: str
    polarizability_rule: str
    thole_direct: float
    thole_mutual: float
    scf_tolerance: float
    scf_max_iterations: int
    scf_nonconvergence: Literal["raise", "warn"]
    d3_parameters: tuple[float, ...]
    neural_cutoff: float
    component_order: tuple[str, ...] = ("elst", "exch", "indu", "disp")
    length_unit: str = "angstrom"
    energy_unit: str = "kcal/mol"
```

Every dataset cache, model checkpoint, training manifest, and prediction result records the `PhysicsConfig` hash. The provider clones charge/multipole inputs before calling existing kernels because the current undamped path may mutate charge tensors. An input-immutability test is mandatory.

## 3. Architectures

### 3.1 A — `MACE-AP3D3-DirectPolar`

- Load `polar-1-s` as a frozen monomer encoder.
- Use PolarMACE density coefficients as candidates for charge and dipole only after explicit convention conversion and validation.
- Predict missing quadrupoles and response properties from MACE hidden irreps using small trainable completion heads.
- Use projected MACE node features for AP3D3 pair residuals.
- Do not instantiate AtomMPNN.

Required property outputs:

```text
q, mu, quadrupole, hfvr, valence_width, alpha, electrostatic_damping
```

A direct PolarMACE tensor is not considered AP3-compatible until charge conservation, axis ordering, Condon–Shortley conversion, units, rotation covariance, and origin behavior pass tests. On the direct-output fixture, direct `q` must equal PolarMACE `charges`, sum to the monomer charge, and intrinsic atomic dipoles plus `qR` must reconstruct the PolarMACE molecular dipole within `1e-5` in the documented output unit.

### 3.2 B — `MACE-AP3D3-H1` and `MACE-AP3D3-H2`

- Load frozen MACE features for AP3D3 atom/pair featurization.
- Retain the existing AtomMPNN/AtomTypeParam property stack for `q`, `mu`, quadrupoles, HFVR, valence widths, polarizabilities, and damping parameters.
- Retain existing pretrained property checkpoints.
- Feed MACE states and the retained atom properties into the AP3D3 pair residual network.
- H1 projects MACE states into AP3D3 `h0` and retains intramonomer updates/directional states.
- H2 bypasses the AP3D3 intramonomer update stack and constructs intermolecular pairs directly from projected MACE states.
- Use the shared long-range provider rather than architecture-specific classical-energy code.

H1 is the first implementation target because it changes representation learning without changing the existing atomic-property source or pair topology. H2 is a separate speed/necessity ablation.

### 3.3 C — `MACE-AP3D3-AtomHead`

- Load frozen MACE monomer features.
- Train a QCMLForge atom-property model on top of MACE irreps:
  - invariant heads: `q`, HFVR, valence width, alpha scale, damping;
  - equivariant `l=1` head: atomic dipoles;
  - equivariant `l=2` head: traceless quadrupoles.
- Use the same MACE features for AP3D3 pair residuals.
- Do not instantiate AtomMPNN.
- Enforce charge conservation, covariance, quadrupole symmetry/tracelessness, and positive response quantities in the model, not only in post-processing.
- If the selected hidden representation lacks usable `l=2` channels, introduce an explicit e3nn tensor-product/product-basis block. Never predict a 3×3 quadrupole with an unconstrained nine-scalar MLP.

C is the preferred full-replacement architecture. A remains a direct-output ablation.

## 4. Shared data contracts

Add typed containers. Exact implementation may use dataclasses, `NamedTuple`, or a compile-safe equivalent, but field names and shapes are normative.

```python
@dataclass
class MACEAtomicFeatures:
    invariant: torch.Tensor        # [n_atom, d_invariant]
    equivariant: torch.Tensor      # [n_atom, d_irreps] or structured irreps
    batch: torch.Tensor            # [n_atom]
    atomic_numbers: torch.Tensor   # [n_atom]
    total_charge: torch.Tensor     # [n_monomer]
    total_spin: torch.Tensor       # [n_monomer], versioned mapping
    feature_schema: str


@dataclass
class AtomicPropertyBundle:
    q: torch.Tensor                # [n_atom, 1]
    mu: torch.Tensor               # [n_atom, 3]
    quadrupole: torch.Tensor       # one canonical representation
    hfvr: torch.Tensor             # [n_atom, 1]
    valence_width: torch.Tensor    # [n_atom, 1]
    alpha: torch.Tensor            # [n_atom, 1]
    damping: torch.Tensor          # [n_atom, 1]


@dataclass
class ClassicalEnergyBundle:
    pair_elst: torch.Tensor
    pair_ind: torch.Tensor
    pair_disp: torch.Tensor
    dimer_elst: torch.Tensor       # [n_dimer]
    dimer_ind: torch.Tensor        # [n_dimer]
    dimer_disp: torch.Tensor       # [n_dimer]
```

Rules:

1. One module owns all conversion between MACE conventions and QCMLForge conventions.
2. The long-range provider accepts `AtomicPropertyBundle`; it must not inspect which architecture produced it.
3. The pair residual core accepts MACE features and atomic properties; it must not compute classical energies.
4. Dimer aggregation occurs once.
5. All public predictions return `[n_dimer, 4]` ordered as ELST, EXCH, IND, DISP.
6. The existing AP3D3 behavior and checkpoint loading must remain covered by regression tests.
7. Dataset conversion/collation propagates monomer charge and multiplicity/spin. Joint-dimer MACE edges are forbidden; a test proves A and B are isolated graphs.
8. Live and precomputed classical paths consume the same `PhysicsConfig` and assemble identical component semantics.
9. Provider inputs remain immutable across long-range evaluation.

For the controlled A/B/C comparison, `polarizability_rule="hfvr-4/3"` is the default: atomic polarizability is derived from the existing free-atom table and predicted HFVR using the repository's current rule. A directly learned `alpha` is a later named ablation, not a silent C-only behavior.

## 5. Proposed code organization

```text
src/apnet_pt/
├── mace/
│   ├── __init__.py                 # lazy exports; no eager optional import
│   ├── schema.py                   # feature/property/physics contracts
│   ├── encoder.py                  # frozen PolarMACE adapter
│   ├── properties.py               # DirectPolar, legacy, and AtomHead providers
│   ├── long_range.py               # one MTP-MTP/induction/D3 spine
│   ├── pair.py                     # H1/H2 feature adapter + residual core
│   └── model.py                    # shared harness and checkpoint reconstruction
├── AtomPairwiseModels/
│   ├── apnet3_d3_fused.py          # minimal injectable seams only
│   └── mtp_mtp.py                  # reuse low-level kernels
├── pt_datasets/
│   └── mace_ap3d3_ds.py
├── training/
│   ├── __init__.py
│   └── mace_ap3d3_factory.py
└── model_io.py

scripts/
├── make_mace_ap3d3_smoke_data.py
└── slurm/
    ├── prepare_mace_ap3d3_features.sbatch
    ├── train_mace_atomic_properties.sbatch
    ├── train_mace_ap3d3.sbatch
    └── submit_mace_ap3d3_matrix.sh

tests/
├── test_mace_polar_adapter.py
├── test_mace_atomic_properties.py
├── test_long_range_sapt.py
├── test_mace_ap3d3_architectures.py
├── test_mace_ap3d3_invariance.py
├── test_mace_ap3d3_cli.py
├── dataset_data/mace_ap3d3_smoke.pkl
└── dataset_data/mace_atomic_properties_smoke.pkl
```

Do not implement three copies of AP3D3 forward logic. Architecture selection belongs in providers/factories; pair construction, residual readouts, long-range assembly, losses, prediction shape, and checkpoint behavior remain shared.

### 5.1 Dependency and environment contract

MACE is optional for the base QCMLForge installation but mandatory for these model options:

```toml
[project.optional-dependencies]
mace = ["mace-torch==<version pinned by M0>"]
```

- M0 records a lockable MACE/e3nn/PyTorch/`graph_electrostatics` combination compatible with `torch>=2.10,<2.11`.
- Recon found candidate `MACE-POLAR-1-S.model` SHA-256 `e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510`; M0 must independently download/verify it before this value becomes normative.
- Add `qcml_dftd3` as an explicit declared dependency at the version used by the existing AP3D3 tests; it is currently imported by production code but absent from `pyproject.toml`.
- `pip install -e '.[mace]'` is the documented local/cluster installation.
- Ordinary CI uses a protocol stub and does not download a foundation model.
- Real integration and cluster environments pin the MACE code version, MACE checkpoint digest, `qcml_dftd3`, CUDA, and e3nn versions in a manifest.
- Dtype conversion is local to model inputs/modules. Do not mutate PyTorch's process-global default dtype as a loader side effect.
## 6. Module interfaces

### 6.1 `MACEPolarFeaturizer`

Responsibilities:

- load a canonical MACE model ID or local checkpoint;
- expose the direct `torch.nn.Module`, not an ASE call per sample;
- run isolated monomer A and B graphs with shared weights;
- set `eval()` and `requires_grad_(False)` by default;
- return a versioned feature schema discovered from the loaded checkpoint; do not hard-code irrep widths from the diagram;
- initially use the complete raw PolarMACE forward and its returned `node_feats`; a faster private-layer tap is allowed later only after parity tests;
- extract invariant features without flattening equivariant channels;
- optionally expose raw irreps for atom-property heads;
- support online and cached modes;
- record model ID, MACE version, checkpoint SHA-256, dtype, supported-element intersection, layer selection, and irreps in checkpoints;
- preserve the existing invalid-dimer behavior while reporting a provider-specific unsupported-element reason;
- never add PolarMACE's own total/electrostatic/interaction energy to SAPT assembly.

The implementation must prefer an explicit PolarMACE output/wrapper over forward hooks. Hooks are not the production API because they are brittle under compilation and upstream changes.

### 6.2 Property providers

Define a common protocol:

```python
class AtomicPropertyProvider(Protocol):
    def forward(
        self,
        batch: DimerBatch,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
    ) -> tuple[AtomicPropertyBundle, AtomicPropertyBundle]: ...
```

Implement:

- `PolarDirectPropertyProvider` for A;
- `LegacyAtomMPNNPropertyProvider` for B;
- `MACEAtomPropertyModel` for C.

### 6.3 `LongRangeSAPTProvider`

```python
class LongRangeSAPTProvider(nn.Module):
    def forward(
        self,
        batch: DimerBatch,
        props_a: AtomicPropertyBundle,
        props_b: AtomicPropertyBundle,
    ) -> ClassicalEnergyBundle: ...
```

Configuration:

```text
electrostatics_mode = damped-cliff | damped-amoeba | undamped
induction_mode      = classical-induction
 dispersion_mode     = d3
```

Default: `damped-cliff` + `classical-induction` + D3.

The provider must call existing low-level functions in `mtp_mtp.py` and `classical_induction.py`. It must not duplicate electrostatics equations. It exposes induction convergence diagnostics and applies the configured non-convergence policy. D3 may enter only through this provider; architecture and pair-residual code may not add a second D3 term.

### 6.4 Pair residual core

Support two MACE integration modes:

- H1: project MACE invariants to `n_embed`, then retain AP3D3 intramonomer updates and directional states;
- H2: bypass AP3D3 intramonomer updates and use projected MACE states directly.

H1 is implemented first. Its projected MACE state replaces the initial atom state; do not also add an extra learned element embedding unless it is a separately named ablation.

Pair inputs include:

```text
MACE atom states, q, HFVR, valence width, radial basis, directional invariants
```

A→B and B→A paths must share weights and sum before dimer aggregation.

## 7. `train_models.py` CLI contract

### 7.1 Canonical pair-model options

```text
MACE-AP3D3-DirectPolar
MACE-AP3D3-H1
MACE-AP3D3-H2
MACE-AP3D3-AtomHead
```

Implement one normalized registry/factory instead of adding four large constructor branches to `train_models.py`. Do not hide H1/H2 behind a public topology flag.

```python
MACE_AP3D3_OPTIONS = {
    "MACE-AP3D3-DirectPolar": {
        "properties": "direct", "pair_mode": "h1", "feature_mode": "all-scalars+norms"
    },
    "MACE-AP3D3-H1": {
        "properties": "legacy", "pair_mode": "h1", "feature_mode": "final-layer-scalars"
    },
    "MACE-AP3D3-H2": {
        "properties": "legacy", "pair_mode": "h2", "feature_mode": "all-scalars+norms"
    },
    "MACE-AP3D3-AtomHead": {
        "properties": "atomhead", "pair_mode": "h1", "feature_mode": "all-scalars+norms"
    },
}
```

### 7.2 New arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--mace_model` | `polar-1-s` | Canonical foundation-model ID |
| `--mace_model_path` | `None` | Optional local checkpoint override |
| `--mace_model_sha256` | `None` | Required digest for cluster runs |
| `--mace_feature_mode` | `auto` | Resolves from the canonical option registry; noncanonical overrides are named ablations |
| `--mace_default_dtype` | `float32` | Backbone inference dtype; validate against checkpoint without changing process-global dtype |
| `--mace_cache_dir` | `None` | Optional architecture/schema-keyed monomer feature cache |
| `--mace_offline` | false | Forbid model download and fail early if the local artifact is unavailable |
| `--mace_atom_model_path` | `None` | DirectPolar/AtomHead atomic-property checkpoint |
| `--mace_property_mode` | `learned` | Atom training only: `direct-completion` or `learned` |
| `--train_atomic_heads` | false | Jointly train A/C property heads; smoke only unless atomic losses are present |
| `--long_range_elst` | `damped-cliff` | `damped-cliff`, `damped-amoeba`, or `undamped` |
| `--d3_params` | repository default | Named preset or serialized parameter file |
| `--smoke_data_path` | `None` | Checked-in qcel dimer + `[N,4]` SAPT label fixture |
| `--smoke_atom_data_path` | `None` | Checked-in monomer + atomic-property target fixture |
| `--skip_compile` | false | Required for smoke/debug runs |
| `--dataloader_num_workers` | current behavior | Set to zero for smoke runs |
| `--overwrite` | false | Permit replacing an existing output checkpoint; otherwise fail |
| `--resume` | false | Resume only from an explicitly compatible checkpoint |

Retain and honor:

```text
--no_disp_nn
--include_total_mse
--use_precomputed_classical / --no-use_precomputed_classical
--unfreeze_dimer_prop_model
--unfreeze_atom_model
--build_dataset_only
--world_size_ddp
--omp_num_threads
```

Validation rules:

- H1/H2 require the existing AtomMPNN/AtomTypeParam checkpoint inputs and reject `--mace_atom_model_path`.
- DirectPolar/AtomHead require `--mace_atom_model_path` for production unless atomic heads are explicitly trained.
- `--train_atomic_heads` without atomic-property labels must print a warning that the run is code-validation only.
- Cluster scripts must provide a local/cached MACE checkpoint and SHA-256; compute nodes must not download it.
- Unknown architecture/flag combinations raise `ValueError` before dataset processing.
- Canonical H1 must resolve to `final-layer-scalars`; canonical H2 must resolve to `all-scalars+norms`. CLI and checkpoint tests assert the resolved schema. Any override changes the run name/manifest to an explicit ablation and may not masquerade as canonical H1/H2.
- `world_size_ddp` must not be overwritten merely because multiple CUDA devices are visible.
- New routes must not inherit the current implicit resume-from-`ap_model_path` behavior. Resume is explicit, and architecture, MACE digest, feature schema, and physics hash must match.
- Constructor arguments may not override checkpoint semantics silently; incompatible overrides fail before model construction.

### 7.3 Optional atom-model training

Add an atom-model entry point for pretraining A/C heads:

```text
--train_am MACE-AtomicProperties
--mace_property_mode direct-completion | learned
```

`direct-completion` trains only Q/response completion around direct PolarMACE q/μ candidates. `learned` trains all C heads.

## 8. Checkpoint contract

Use `model_io` versioned checkpoints. Do not pickle an ASE calculator.

Required metadata:

```text
model_type = MACEAP3D3
architecture = direct-polar | hybrid-h1 | hybrid-h2 | atomhead
mace_model_id
mace_version
mace_checkpoint_sha256
mace_feature_schema
mace_feature_mode
pair_mode
mace_default_dtype
atomic_property_schema
long_range_elst
long_range_induction = classical-induction
d3_parameters
component_order = [elst, exch, indu, disp]
units
physics_config_hash
dataset_manifest_hash
preprocessing_schema
split_id_hash
training_seed
parameter_counts
source_commit
```

Submodels:

- feature adapter;
- property provider;
- pair residual core;
- long-range configuration;
- legacy AtomMPNN/AtomTypeParam checkpoint for B;
- atom-property head for A/C.

Do not embed or redistribute the ASL-licensed MACE foundation checkpoint by default. Store its canonical locator, digest, license acknowledgment, and source version and fail clearly if the resolved artifact does not match.

Current `model_io.create_checkpoint()` serializes the complete module `state_dict()`. M6 must add an explicit external-backbone mechanism rather than relying on `requires_grad=False`:

1. checkpoint creation filters exactly the registered MACE backbone key prefixes and stores an `external_submodels.mace` artifact record;
2. the saved checkpoint contains no MACE backbone tensors; a test scans all state-dict keys and serialized tensor sizes;
3. loading resolves the user/local MACE artifact, verifies SHA-256 and model class, constructs the backbone, then loads QCMLForge state;
4. `load_state_dict` may report missing keys only under the declared MACE prefixes; every other missing or unexpected key is fatal;
5. reconstruction verifies architecture, feature schema, dtype policy, physics hash, and route-specific submodel digests before inference;
6. legacy v1/v2 QCMLForge checkpoints continue to load unchanged.

Save/reload tests must prove prediction equality on the smoke fixture and prove that moving/removing the external MACE artifact produces a clear digest/path error.

## 9. Feature cache contract

Frozen features may be cached after online-mode correctness is established.

Cache key fields:

```text
MACE checkpoint digest
MACE version
feature schema
atomic numbers
coordinates in angstrom
fragment charge
spin convention/value
dtype
atom order
```

Rules:

- Cache isolated monomers, not full dimers.
- A/C caches must retain the raw irreps required by property heads.
- H1/H2 may cache only the invariant pair schema if raw irreps are not used.
- Cache and online predictions must agree within dtype-specific tolerance.
- Cache invalidation is mandatory when any key field changes.
- Equivariant cached tensors may not be reused after rotating/reorienting coordinates unless the cache applies the exact corresponding irrep transform. The initial implementation treats transformed coordinates as a cache miss.
- Coordinate quantization, if any, is part of the schema and must not merge geometries beyond the tested tolerance.

## 10. Small-data verification

### 10.1 Checked-in fixtures

Create `tests/dataset_data/mace_ap3d3_smoke.pkl` containing 8–16 qcelemental dimers and `[N,4]` SAPT0 labels. Reuse existing test molecules and labels where possible.

Create `tests/dataset_data/mace_atomic_properties_smoke.pkl` containing a small monomer set with the atomic targets needed to exercise DirectPolar completion and AtomHead training. Targets must declare units, quadrupole convention, source/provenance, and which fields are true references versus deterministic wiring-only fixtures.

Minimum coverage:

- at least two chemical/geometric cases;
- at least one close-range geometry;
- at least one long-range geometry;
- neutral fragments; add an ionic/radical case only after MACE charge/spin conventions are verified;
- deterministic order and split.

This fixture verifies execution, not model quality.

### 10.2 CI tests without network access

Use a tiny deterministic MACE protocol stub for ordinary CI. Mark real-checkpoint tests separately:

```python
@pytest.mark.mace_integration
```

CI unit tests must cover:

- feature and property shapes;
- backbone gradients remain absent;
- rotations, translations, atom permutations, A/B swap, and batch-order equivalence;
- charge sum, dipole/quadrupole covariance, and quadrupole symmetry/tracelessness;
- positive/finite response quantities;
- CLI dispatch for all four canonical names;
- finite MTP-MTP, induction, and D3 terms for every architecture;
- all electrostatics modes;
- evaluator input immutability;
- induction convergence and configured non-convergence behavior;
- no D3 double counting;
- prediction shape `[N,4]` with and without `--no_disp_nn`;
- one-epoch training, checkpoint write, reload, and deterministic prediction;
- online/cache parity and cache invalidation;
- separation scans beyond `r_cut_im`, neural cutoff continuity, and correct long-range decay;
- isolated-monomer feature graphs with no A–B MACE edges;
- legacy AP3D3 regression tests.

Wiring tests may compare a wrapper with the low-level function it calls, but those tests do not establish scientific correctness. Check in independent numeric reference values for at least one CLIFF, AMOEBA, undamped, induction, and D3 case. Expected values must come from an external/reference artifact or a previously validated fixture, not from the function under test in the same test run.

Tolerance policy is stored with each fixture. Initial defaults are:

```text
float64 symmetry/cache checks: atol 1e-7, rtol 1e-7
float32 symmetry/cache checks: atol 2e-4, rtol 2e-4
checkpoint reload:            exact config/hash; numeric tolerance by dtype
physics references:           fixture-specific documented tolerance
```

### 10.3 Real-checkpoint local smoke commands

First produce A/C atom-head checkpoints with one-epoch smoke training:

```bash
for MODE in direct-completion learned; do
  python train_models.py \
    --train_am MACE-AtomicProperties \
    --mace_property_mode "$MODE" \
    --mace_model polar-1-s \
    --mace_model_path "$MACE_POLAR_1S" \
    --mace_model_sha256 "$MACE_POLAR_1S_SHA256" \
    --mace_offline \
    --smoke_atom_data_path tests/dataset_data/mace_atomic_properties_smoke.pkl \
    --n_epochs_atom 1 \
    --world_size_ddp 1 \
    --skip_compile \
    --am_model_path "agent_scratch/models/mace-${MODE}.pt"
done
```

Then run all four MACE pair-model options and the matched existing AP3D3 baseline. The final implementation must make commands of this form work:

```bash
COMMON=(
  --mace_model polar-1-s
  --mace_model_path "$MACE_POLAR_1S"
  --mace_model_sha256 "$MACE_POLAR_1S_SHA256"
  --mace_offline
  --mace_default_dtype float32
  --mace_cache_dir agent_scratch/mace_ap3d3_smoke/mace-cache
  --smoke_data_path tests/dataset_data/mace_ap3d3_smoke.pkl
  --data_dir agent_scratch/mace_ap3d3_smoke
  --ds_max_size 8
  --n_epochs 1
  --lr 5e-4
  --world_size_ddp 1
  --omp_num_threads 2
  --dataloader_num_workers 0
  --skip_compile
  --long_range_elst damped-cliff
  --include_total_mse
)

for ARCH in MACE-AP3D3-H1 MACE-AP3D3-H2; do
  python train_models.py --train_apnet "$ARCH" \
    --am_model_path tests/test_models/ap3_ensemble_0/am_3.pt \
    --atom_type_param_model_path tests/test_models/ap3_ensemble_0/am_h+1_3.pt \
    --atom_type_param_model_path2 tests/test_models/ap3_ensemble_0/am_elst_h+1_3.pt \
    --ap_model_path "agent_scratch/models/${ARCH}.pt" \
    "${COMMON[@]}"
done

python train_models.py --train_apnet MACE-AP3D3-DirectPolar \
  --mace_atom_model_path agent_scratch/models/mace-direct-completion.pt \
  --ap_model_path agent_scratch/models/mace_ap3d3_direct.pt \
  "${COMMON[@]}"

python train_models.py --train_apnet MACE-AP3D3-AtomHead \
  --mace_atom_model_path agent_scratch/models/mace-learned.pt \
  --ap_model_path agent_scratch/models/mace_ap3d3_atomhead.pt \
  "${COMMON[@]}"

python train_models.py --train_apnet APNet3-fused-d3 \
  --am_model_path tests/test_models/ap3_ensemble_0/am_3.pt \
  --atom_type_param_model_path tests/test_models/ap3_ensemble_0/am_h+1_3.pt \
  --atom_type_param_model_path2 tests/test_models/ap3_ensemble_0/am_elst_h+1_3.pt \
  --smoke_data_path tests/dataset_data/mace_ap3d3_smoke.pkl \
  --data_dir agent_scratch/mace_ap3d3_smoke/base \
  --ds_max_size 8 --n_epochs 1 --lr 5e-4 \
  --world_size_ddp 1 --omp_num_threads 2 --dataloader_num_workers 0 \
  --skip_compile --long_range_elst damped-cliff --include_total_mse \
  --ap_model_path agent_scratch/models/ap3d3_base.pt
```

Smoke pass criteria for each command:

1. exit code zero;
2. one epoch completes;
3. loss and all component terms are finite;
4. output checkpoint exists;
5. reload succeeds;
6. prediction shape is `[N,4]`;
7. the MACE backbone has no gradients for the four MACE options;
8. logged classical ELST, IND, and DISP are present;
9. a long-range example retains classical asymptotics while NN residuals switch off smoothly.

## 11. Worktree and subagent execution plan

### 11.1 Preconditions

The current worktree contains untracked documentation. Before launching writer worktrees:

```bash
git switch mace-apnet
git add docs/mace-apnet.html docs/mace-apnet-implementation-spec.md
git commit -m "docs: specify MACE AP3D3 architectures"

# Move/delete generated context-build or reviewer artifacts intentionally;
# do not use a destructive blanket git clean.
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  echo "Refusing to launch writer worktrees from a dirty base" >&2
  exit 1
fi
```

Subagent output artifacts must resolve outside the repository or under ignored `agent_scratch/`. `subagent(..., worktree: true)` requires the clean-base check above to pass.

### 11.2 Milestones and ownership

| ID | Milestone | Writer ownership | Depends on |
|---|---|---|---|
| M0 | Commit atlas/spec; pin base SHA, external artifacts, license, and environment | integration owner | — |
| M1 | Isolated MACE feasibility probe; no production edits | one spike worker | M0 |
| M2 | Immutable `PhysicsConfig`, state/property bundles, reference fixtures, long-range provider | one core worker | M1 |
| M3 | Frozen MACE feature interface, isolated-monomer dataset fields, shared pair seams, stub tests | one core worker | M2 |
| M4B | Hybrid H1 provider/topology + focused tests | isolated worker/worktree | M3 |
| M4C | Learned AtomHead C + focused tests | isolated worker/worktree | M3 |
| M5B | Hybrid H2 ablation using the merged H1 provider | isolated worker/worktree | M4B |
| M5A | DirectPolar q/μ conversion; reuse C's Q/response modules | isolated worker/worktree | M4C |
| M6 | Merge routes into shared harness; registry, checkpoint, and `train_models.py` wiring | one integration worker | M4B/C, M5A/B |
| M7 | Checked-in atom/dimer fixtures, two atom-head smokes, four MACE pair smokes, matched BASE smoke | one integration worker | M6 |
| M8 | SLURM feature-precompute, atom-head, and pair-training scripts | one worker | M7 pass |
| M9 | Parallel scientific/software review, one fix worker, focused re-review | reviewers + one fix worker | M8 |

Only M4B/M4C and later M5A/M5B may use concurrent writer worktrees. A depends on C so that DirectPolar reuses the same quadrupole/response modules; otherwise A is not a clean direct-q/μ ablation. Shared files such as `train_models.py`, `model_io.py`, `mace/long_range.py`, `mace/model.py`, registry code, and dataset factories always have one writer.

### 11.3 Suggested branches/worktrees

```text
mace-apnet                 integration branch
mace-apnet-spike           M1; disposable evidence branch
mace-apnet-core            M2/M3
mace-apnet-h1              M4B
mace-apnet-atomhead        M4C
mace-apnet-h2              M5B, based on merged H1
mace-apnet-direct          M5A, based on merged AtomHead
mace-apnet-cli-cluster     M6/M7/M8 after route merge
```

Merge route commits sequentially into `mace-apnet`. Route branches add only route-specific modules and uniquely named tests; they provide registry snippets in handoff notes instead of editing shared registration files.

Every worktree/run receives unique values for `data_dir`, model output, `TMPDIR`, MACE/Hugging Face cache, feature cache, and diagnostic directory. Raw data are shared read-only. Concurrent workers must never process or write the same LMDB/cache root.

### 11.4 Subagent roles

- `context-builder`/`scout`: read-only repository mapping before each milestone.
- `worker`: the sole writer for a milestone/worktree.
- `reviewer`: fresh-context correctness, scientific-contract, and test review.
- `oracle`: architecture escalation only when the approved contracts cannot be implemented without changing scope.
- parent/orchestrator: owns merge order, accepted decisions, and final validation.

Every implementation worker receives structured acceptance criteria. Example first route fan-out after M3 is merged and the integration worktree is clean:

```javascript
subagent({
  tasks: [
    {
      agent: "worker",
      task: "Implement Hybrid H1 from docs/mace-apnet-implementation-spec.md. Edit only the legacy provider/topology module and focused tests; do not edit train_models.py, registry, dataset factory, or shared long-range code.",
      acceptance: {
        criteria: [
          "H1 returns canonical feature/property bundles",
          "Legacy property predictions remain within tolerance",
          "MACE initializes h0 and AP3D3 intra updates remain active",
          "Focused shape, symmetry, finite-energy, and checkpoint tests pass"
        ],
        evidence: ["changed-files", "tests-added", "commands-run", "validation-output", "residual-risks"],
        stopRules: ["Do not duplicate AP3D3 or long-range forward logic", "Escalate convention ambiguity"],
        maxFinalizationTurns: 3
      }
    },
    {
      agent: "worker",
      task: "Implement AtomHead C from the spec. Edit only C's equivariant/invariant heads and focused tests; do not edit train_models.py, registry, dataset factory, or shared long-range code."
    }
  ],
  concurrency: 2,
  worktree: true,
  context: "fresh"
})
```

After H1 and C merge, launch a second two-worktree fan-out for H2 (based on H1) and DirectPolar (based on C). Then use one integration worker for CLI wiring, followed by fresh parallel reviewers with these angles:

1. tensor shapes, conventions, symmetry, and checkpoint correctness;
2. MTP-MTP/induction/D3 asymptotics and double-counting;
3. dataset, CLI, smoke commands, and cluster reproducibility;
4. performance, feature caching, and memory.

Accepted fixes are applied by one fix worker, followed by focused re-review.

## 12. SLURM scale-up

Do not submit full training until unit tests, all four MACE option smoke runs, and the matched AP3D3 baseline smoke run pass.

### 12.1 Prepare job

`prepare_mace_ap3d3_features.sbatch` must:

- run on a login-approved or compute partition with no hidden downloads;
- verify the MACE checkpoint SHA-256;
- build/reuse monomer feature caches;
- write a manifest containing source commit, MACE version/digest, feature schema, dataset identity, and counts;
- exit nonzero on partial cache generation;
- support restart without overwriting valid entries.

### 12.2 Training job

`train_mace_ap3d3.sbatch` must accept environment variables or positional parameters for:

```text
MODEL_OPTION
SEED
DATA_DIR
FEATURE_CACHE_DIR
MODEL_OUT
MACE_MODEL_PATH
MACE_MODEL_SHA256
AM_MODEL_PATH
AM_MODEL_SHA256
ATOM_TYPE_PARAM_MODEL_PATH
ATOM_TYPE_PARAM_MODEL_SHA256
ATOM_TYPE_PARAM_MODEL_PATH2
ATOM_TYPE_PARAM_MODEL_SHA256_2
MACE_ATOM_MODEL_PATH
MACE_ATOM_MODEL_SHA256
PHYSICS_CONFIG_PATH
PHYSICS_CONFIG_SHA256
ELECTROSTATICS_MODE
N_EPOCHS
BATCH_SIZE
```

Requirements:

- `set -euo pipefail`;
- one GPU per task for the initial scale-up;
- `srun python -u train_models.py ...`;
- logs, caches, diagnostics, `TMPDIR`, processed dataset root, and checkpoints under a unique run directory;
- environment, git commit, dataset manifest, immutable split IDs, preprocessing schema, and physics hash captured before training;
- SIGTERM/requeue-safe checkpointing if supported;
- no implicit resume from an unrelated existing `ap_model_path`;
- validate only the route-specific checkpoint inputs: H1/H2/BASE require all three legacy paths/digests; DirectPolar/AtomHead require the MACE atom-property path/digest; BASE does not require MACE;
- include every consumed submodel path and digest in the run manifest;
- fail if the output path contains a checkpoint with a mismatched architecture, MACE digest, submodel digest, feature schema, split hash, or physics hash;
- use exactly one `srun` task for the current internal-spawn training path; do not combine `--ntasks-per-node=4` with internal multiprocessing.

### 12.3 Submission matrix

`submit_mace_ap3d3_matrix.sh` should submit:

```text
5 controlled experiments (BASE, H1, H2, DirectPolar, AtomHead) × 3 seeds × selected electrostatics mode
```

Recommended staged scale:

1. 32–256 dimers, 2 epochs, one GPU;
2. 1k–10k dimers, 5 epochs, one GPU;
3. full data and production epochs;
4. multi-GPU only after a dedicated single-GPU/DDP parity test and a decision to retain internal spawn or move to `torchrun`.

Use `afterok` dependencies from feature preparation to training. `train_mace_atomic_properties.sbatch` produces DirectPolar-completion and AtomHead checkpoints after feature preparation; DirectPolar/AtomHead pair jobs depend on those successful jobs. H1/H2/BASE depend on validated legacy checkpoint staging. A failed smoke, preparation, or atom-head job must not release dependent pair training jobs.

### 12.4 Cluster acceptance

Before the full-data submission, one small `sbatch` job per controlled experiment, including BASE, must prove:

- correct model option printed in the manifest;
- correct foundation and route-specific submodel digests;
- feature cache loads where applicable;
- one or two epochs finish;
- checkpoint reload and prediction run on the compute node;
- no NaN/Inf in features, multipoles, classical components, residuals, loss, or gradients;
- memory and elapsed time are recorded;
- no network access was required.

## 13. Test and review commands

Minimum focused suite:

```bash
python -m pytest \
  tests/test_mace_polar_adapter.py \
  tests/test_mace_atomic_properties.py \
  tests/test_long_range_sapt.py \
  tests/test_mace_ap3d3_architectures.py \
  tests/test_mace_ap3d3_invariance.py \
  tests/test_mace_ap3d3_cli.py -v

python -m pytest \
  tests/test_ap3_d3_fused.py \
  tests/test_classical_components.py -k "mtp or damping or induction or d3 or ap3" -v

git diff --check
```

Real MACE integration:

```bash
python -m pytest -m mace_integration -v
```

Full suite is required before cluster submission:

```bash
python -m pytest tests/
```

## 14. Acceptance criteria

### Shared

- [ ] `polar-1-s` can be loaded from a pinned local artifact.
- [ ] Frozen MACE features are produced for isolated monomers.
- [ ] No scalar MLP receives flattened covariant channels.
- [ ] One shared long-range provider implements MTP-MTP, `classical_induction`, and D3.
- [ ] CLIFF, AMOEBA, and undamped electrostatics modes are explicit and tested against independent references.
- [ ] `PhysicsConfig` is immutable, hashed, and identical across controlled comparisons.
- [ ] MTP-MTP/induction inputs remain unchanged after evaluation.
- [ ] D3 is added exactly once.
- [ ] All predictions are `[N,4]` in kcal/mol.
- [ ] Existing AP3D3 tests and checkpoint loading remain valid.
- [ ] Saved MACE-model checkpoints contain no foundation-backbone tensors and reconstruct only from a digest-verified external artifact.

### A

- [ ] Direct q/μ conversion is documented and tested.
- [ ] Q and response completion heads exist.
- [ ] AtomMPNN is absent.
- [ ] All physical constraints pass.

### B / H1 / H2

- [ ] Existing atom-property predictions are unchanged within tolerance.
- [ ] MACE features enter both pair paths.
- [ ] H1 demonstrably retains intramonomer/directional updates and resolves `final-layer-scalars`.
- [ ] H2 demonstrably bypasses those updates, resolves `all-scalars+norms`, and remains a distinct checkpoint/CLI architecture.
- [ ] Existing property checkpoints are portable in the new checkpoint.

### C

- [ ] Invariant and equivariant heads emit the complete property bundle.
- [ ] Charge conservation, covariance, tracelessness, and positivity pass.
- [ ] AtomMPNN is absent.
- [ ] Atom-property checkpoint pretraining and loading work.

### End-to-end

- [ ] Every canonical MACE `train_models.py` option and the matched AP3D3 BASE complete the small-data smoke run on identical split IDs and physics settings.
- [ ] Both A/C atom-head pretraining smoke commands produce the checkpoints consumed by pair smoke runs.
- [ ] Every checkpoint reloads and reproduces predictions.
- [ ] Every model option logs separate classical and residual component terms plus induction convergence diagnostics.
- [ ] Small `sbatch` verification jobs pass for BASE and all four MACE options before full-scale submission.
- [ ] Full-data scripts require an explicit model option, seed, checkpoint digest, split hash, physics hash, and output directory.

## 15. Stop rules and unresolved decisions

Stop implementation and escalate rather than guessing if:

- the loaded PolarMACE checkpoint does not expose a stable local node representation;
- MACE spin conventions cannot be mapped unambiguously from qcelemental metadata;
- direct density coefficients cannot be converted reproducibly to QCMLForge q/μ conventions;
- the requested undamped or AMOEBA mode would require silently substituting CLIFF damping;
- a shared refactor would break legacy AP3D3 checkpoint compatibility;
- MACE and the repository's PyTorch/e3nn versions cannot coexist in one locked environment;
- ASL checkpoint terms prevent the intended cluster caching or redistribution approach.

Decisions to record during M0/M1:

1. pinned `mace-torch` version and exact checkpoint digest;
2. MACE spin/multiplicity mapping;
3. canonical quadrupole representation and units;
4. fixed pair-head capacity and feature schema for the controlled H1/H2/A/C comparison;
5. atomic-property supervision source and model-selection thresholds for A/C;
6. exact D3 parameter preset per SAPT target level;
7. whether real MACE integration tests run in default CI or a dedicated job.

## 16. Definition of done

The branch is ready for cluster-scale science only when all three property architectures and all four exact CLI options are implemented, share the same tested asymptotic provider, pass unit and real-checkpoint smoke tests, save/reload reproducibly, and complete a small `sbatch` run with recorded manifests. Successful construction alone is not sufficient; finite one-epoch training and inference evidence is required for DirectPolar, H1, H2, and AtomHead.
