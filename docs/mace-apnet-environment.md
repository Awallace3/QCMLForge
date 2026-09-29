# MACE/AP3D3 environment and external-artifact policy

The MACE routes are optional. Base QCMLForge installations and imports do not
import MACE. Install the pinned CPU-compatible stack with:

```bash
pip install -e '.[mace]'
```

The validated dependency lock is Python 3.12, PyTorch `>=2.10,<2.11`,
`mace-torch==0.3.16`, `e3nn==0.4.4`, and `graph-longrange` at commit
`0e21d5546c482d08388a08eb4d948e833227ce47` (tag v0.4.0). CUDA wheel,
driver, memory, and CPU/GPU parity remain cluster preflight requirements.
QCMLForge's `qcml_dftd3` implementation is vendored and is the sole D3
implementation used by the shared long-range provider.

## PolarMACE artifact

The canonical model ID is `polar-1-s`. The external artifact is:

- file: `MACE-POLAR-1-S.model`
- source: <https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model>
- size: 33,375,439 bytes
- SHA-256: `e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510`
- checkpoint license: Academic Software License (ASL)

Never deserialize a downloaded model before comparing its exact size and
SHA-256 with the trusted manifest. Protected tests resolve the administrator
mounted path through `QCMLFORGE_POLARMACE_ARTIFACT`; missing protected
prerequisites are `BLOCKED`, while wrong size/digest is `FAIL`. The Python
preflight checks Unix permission bits only and does not prove mount immutability.
CI-2 is currently **DISABLED/BLOCKED** by an unconditional workflow guard. The
three confirmed external blockers are: no `polarmace-protected` GitHub
Environment, unprotected `main`, and no isolated matching self-hosted runner.
Before a reviewed future change enables it, administrators must verify protected
Environment restrictions and required reviewers, branch protection, a
read-only approved mount, and isolated runner cleanup. The future authority requires
`QCMLFORGE_CI2_POLICY_ATTESTED=true` from that protected Environment; missing or
false is `BLOCKED`. A local environment value or repository YAML is not proof of
external policy. Trusted-ref YAML remains defense in depth, not the artifact
security boundary. Cluster jobs must use offline mode, a local path, and an
explicit digest; compute nodes must
not download models. The artifact remains external: do not commit, package,
embed in QCMLForge checkpoints, copy into caches/evidence, upload, or
redistribute it. Checkpoints and run manifests record its canonical locator,
digest, MACE version, schema, and license acknowledgment.

The ASL checkpoint is intended here only for approved academic,
non-commercial internal use. Commercial use, external collaboration,
redistribution, or modified checkpoint publication requires institutional
license review. MACE and graph-longrange source code are separately MIT
licensed.

## Conventions fixed for the first implementation

- coordinates: angstrom
- energies: kcal/mol
- public component order: `elst, exch, indu, disp`
- quadrupoles: Cartesian symmetric-traceless `[n_atom, 3, 3]`
- MACE input `total_spin`: QCElemental molecular multiplicity represented as a
  floating tensor; the PolarMACE physical spin constraint uses multiplicity
  minus one internally
- atomic polarizability: existing positive `abs(HFVR)`-to-free-atom `4/3` rule
- CLIFF electrostatic damping: the existing positive `abs(damping)` convention;
  AMOEBA uses separate explicit positive K inputs and never re-labels CLIFF K
- induction: existing AP3 Thole self-consistent induced-dipole kernel, with
  explicit direct and mutual controls (defaults `0.34` and `0.39`); legacy
  callers using the single `0.39` argument retain identical behavior
- full pair semantics: every intermonomer Cartesian atom pair exactly once
- D3: vendored SAPT(PBE0)-D3(I) default unless an immutable physics config
  supplies `(s6, s8, a1, a2)` explicitly

Inactive electrostatics parameter tuples, alternate polarizability rules, and
cutoff-only full-edge semantics are rejected rather than merely hashed.

The verified `polar-1-s` public `node_feats` schema is 512 scalar channels.
The reviewed MACE `0.3.16` private adapter exposes the interaction output as
`512x0e+512x1o+512x2e+512x3o`, converts MACE's channel-major storage to
standard e3nn irrep-major storage, and must reproduce the public final 512
scalars within `1e-6` on every use. A version, class, layout, or parity mismatch
is fatal; there is no silent private-feature fallback.

Direct outputs use the artifact contract
`polar-density-l1-yzx-eangstrom-v1`: coefficient zero is atomic charge and
coefficients `[3, 1, 2]` are intrinsic Cartesian x/y/z dipoles in eÅ. Atomic
dipoles are converted to e·bohr at the QCMLForge property seam. Intrinsic
atomic dipoles plus `qR` must reconstruct the public molecular dipole within
`1e-5` eÅ. Passing these wiring and covariance checks does not establish
scientific accuracy for DirectPolar quadrupole/response completion heads.
DirectPolar charges themselves remain the unchanged public PolarMACE values;
per-monomer sums must agree with both PolarMACE and QCMLForge total-charge
records, and inconsistent outputs are rejected rather than projected.

The neural residual support is the AP3 intermonomer cutoff `r_cut_im=8.0 Å`.
It is part of `PhysicsConfig`, feature-cache identity, and checkpoint semantics.
Checked separation scans establish numerical classical decay and energy/first-
derivative continuity at that cutoff only for the tested kernels; they are not
external or deployment validation.

`damped-amoeba` requires explicit per-atom AMOEBA/HIPPO K inputs. The checked
`amoeba_water_dimer_ref.pkl` path uses its `alpha_A`/`alpha_B` values and the
total-atomic-monopole q convention (the kernel subtracts nuclear Z). The
production-kernel value does not reproduce the fixture's independent HIPPO
energy, so CLI routes without mode-specific inputs fail early and no AMOEBA
scientific-equivalence claim is made.

Precomputed classical ledgers must be complete `ClassicalEnergyBundle` records
carrying the exact active `PhysicsConfig` hash. Datasets without matching
ledgers fail before construction. MACE pair smoke reports and diagnostics
persist induction convergence, iterations, residual, and policy; those fields
do not establish broader model accuracy.

## Small SLURM verification

The offline preparation, atomic-head, pair-matrix, manifest, and dependency
workflow is documented in [mace-apnet-slurm.md](mace-apnet-slurm.md). Checked-in
scripts, shell parsing, and fake-`sbatch` tests validate local wiring only; no
real scheduler validation has been performed by those checks. Run the local
smoke and dry-run gates before any separately authorized `sbatch`. Full-scale
and multi-GPU runs remain blocked on successful real small jobs and the
outstanding CUDA preflight.
