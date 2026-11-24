# Goal: Ewald/PME long range for FF-2 in torch ("torch-pes")

**Status:** Goal (design open, nothing implemented)
**Scope:** `apnet_pt.ff2` energy path used for MD (`ff2/md.py`, `ff2/physics.py`, `ff2/model.py`)
**Opened:** 2026-11-20

## 1. Goal

Make FF-2's cost grow close to linearly with system size, and make it usable on
periodic, condensed-phase systems. To do this, split every long-range term into
two parts, as SO3LR-T-M does with jax-pme:

- a short-range part, evaluated in real space on a neighbour list;
- a long-range part, evaluated in reciprocal space by particle-mesh Ewald.

The terms affected are electrostatics, polarization and dispersion. All of it
stays in torch: differentiable, compile-friendly and runnable on GPU. Forces
keep coming from autograd.

## 2. Why

FF-2 currently treats a whole MD system as one FF-2 "dimer" (`md.cluster_batch`):

- every pair of atoms on different monomers is a pair term, so the pair list is
  O(N²);
- polarization builds one dense `(3m, 3m)` system over all atoms
  (`physics.polarization_system`). It is solved by Cholesky, which costs O(N³)
  time and O(N²) memory, or by block-Jacobi CG on the same dense matrix.

The perf tracker shows where this breaks. FF-2's fastest accurate setting
(2D table + compile, float64), measured against SO3LR-T-M on the same hardware:

| Hardware | Up to ~200 atoms | 384 benzene atoms | 768 benzene atoms |
|---|---|---|---|
| One CPU core | 0.85–2.7× SO3LR-T-M's time | 2.3× | 4.6× |
| One V100 | 3.6–5.4× | 20× | 46× (463 ms vs 10 ms) |

As benzene clusters double in size, SO3LR's time barely changes and FF-2's grows
about 4×.

Sources:
- `qcmlforge-exp` `analysis/ff2-perf/perf-pt215.json`, `perf-gpu.json`;
- https://cos-4a16505.tail7d58ac.ts.net/docs/qcml/dynamics/ff2-perf-2026-11-19.html

## 3. What SO3LR actually does (read from `so3lr-v5` @ 0b590a1)

- **No box** (every cluster timing above): long range is real space only, up to
  `lr_cutoff = 12 Å`. Coulomb uses an erf-damped, shifted-force,
  smoothly-switched kernel (`coulomb_erf_shifted_force_smooth`), and dispersion
  has its own long-range damping. **None of SO3LR's cluster advantage comes from
  PME.** It comes from the cutoff, the sparse neighbour list, and fused GPU
  kernels (`so3lr.flash`).
- **Periodic box:** electrostatics use `kspace_electrostatics` set to `ewald`,
  `pme` or `p3m`, via `jaxpme` (jax-pme 0.1.0a1). Dispersion can optionally use
  `kspace_dispersion` set to `pme` or `ewald`, which is r⁻⁶ in k-space. Its
  parameter tuner (`so3lr/cli/tune_ewald.py`) follows torch-pme's tuner. torch-pme
  is the torch sibling of jax-pme, from the same group (lab-cosmo).

So the goal has two parts, and the first one matters even without PME:

1. **Cutoff + neighbour list** for non-periodic systems and for the real-space
   half of Ewald. This alone should flatten the cluster curve.
2. **PME** for periodic boxes: liquids, solvated systems, NPT.

## 4. Why this needs thought: FF-2 is not point charges

SO3LR's long range is point charges plus pairwise C6. FF-2's is not, and each
term needs its own split.

### 4.1 Electrostatics

FF-2 places charges (nucleus, frozen core and Slater cloud), dipoles and
quadrupoles on each atom, in two shells. Pairs are computed with two-centre
Slater kernels (`elst_pairs`).

Beyond `beta_min R > X_SWITCH`, `by_range` already switches to the
point-multipole limit (perf task 1). That gives a natural split:

```
E_elst = Σ_{r<rc} [E_Slater(r) − E_point(r)]        (exponentially short-ranged)
       + E_point-multipole^Ewald                     (all pairs, PME up to quadrupoles)
       − Σ_{same monomer} E_point(r) − self terms    (exclusion corrections)
```

Open questions:
- PME needs multipoles up to rank 2, as in AMOEBA (Sagui–Darden), not just
  charges. Check whether torch-pme handles dipoles and quadrupoles, or whether
  we write a multipolar B-spline spreading step.
- The charges come from a per-monomer chemical-potential solve on each geometry,
  so dq/dR must flow through the mesh. Autograd handles this if the PME is
  written in torch.
- FF-2 elst is intermolecular only, so every same-monomer pair must be removed
  from the reciprocal sum. The exclusion lists are static, from
  `md.cluster_batch`.

### 4.2 Polarization

Polarization is mutual induction with Slater-damped dipole tensors (`l33`, `l53`).
It is whitened and variational, and uses the envelope theorem for forces.

- The dense Cholesky solve must become iterative CG. Each matvec is the damped
  real-space part (sparse, on the neighbour list) plus a PME dipole–dipole
  contribution.
- The variational form keeps forces exact only at convergence. The CG tolerance
  therefore becomes a force-accuracy knob that needs a test.
- Precondition with the existing per-atom block-Jacobi.
- This is the term most likely to dominate cost. Each CG step needs one PME
  dipole spread and gather.

### 4.3 Dispersion

Dispersion is BJ-damped C6 and C8. The C6 term uses an environment-dependent
Casimir–Polder combination, `C6_ij = 2 a_i a_j η_i η_j / (η_i + η_j)`.

- The short-range part, `E_BJ − (−C6/r⁶)`, decays as r⁻¹². C8/r⁸ can be cut off
  with a tail correction.
- r⁻⁶ PME (LJPME) needs a factorizable combination rule. Casimir–Polder is not
  one. Options:
  - a geometric-mean approximation in k-space, with the exact rule in real space
    (what OpenMM's LJPME does);
  - a low-rank expansion of the `η_i η_j / (η_i + η_j)` kernel, with one mesh per
    rank;
  - a cutoff plus a homogeneous tail correction only.

  Measure the error of each before choosing.

### 4.4 Exchange, charge transfer, trunk

These terms are already exponentially short-ranged, with `s_cut` windowing for
exchange and CT. They only need the neighbour list.

## 5. Shared engineering constraints

- **Compile-friendly neighbour list:** a cell list with fixed capacity and a
  skin (jax-md style), so shapes stay static between rebuilds. Rebuild only when
  displacement exceeds skin/2. No graph breaks, per AGENTS.md.
- **No cluster-sized dense tensors** anywhere in the MD path: no dense
  polarization matrix, and no all-pairs `triu` lists.
- **Cluster parity:** for a small system in a large box, or with no box and a
  large cutoff, the energy must reproduce today's exact all-pairs FF-2 to
  ≤1e-6 kJ/mol (float64) and forces to within a tolerance stated up front.
- **Reuse the existing kernel machinery** for the real-space residuals
  (`table2d`, `point`, `by_range`).
- **Dependencies:** prefer torch-pme as a dependency if it covers what we need
  (charges, r⁻⁶, per-channel weights). Otherwise write a minimal B-spline PME
  module in `apnet_pt` and validate it against torch-pme or direct Ewald.
- **Parameter tuning:** set Ewald α, mesh spacing and spline order from an error
  bound, as torch-pme and SO3LR do, not by hand.

## 6. Suggested order

1. **Cutoff + neighbour list, no PME.**
   - Add a real-space cutoff with shifted-force or erf-damped point tails, plus
     a sparse polarization matvec.
   - Measure accuracy against all-pairs FF-2 and scaling to 768 atoms and
     beyond, on CPU and V100.
   - This tests whether the cluster gap to SO3LR closes without any mesh.
2. **Point-charge Ewald/PME** for the monopole part. Validate on a periodic
   water box against direct Ewald.
3. **Multipolar PME** (dipoles, quadrupoles) and the exclusion corrections.
4. **PME inside the CG polarization matvec.**
5. **Dispersion long range:** compare the options in §4.3 and choose one.
6. **OpenMM periodic MD** through PythonForce (box vectors passed per step):
   NVE energy drift, NPT density on a liquid, and timing against SO3LR-T-M with
   `kspace_electrostatics="pme"`.

## 7. Success criteria

- **Scaling:** FF-2 time per energy+forces call grows ≤ ~1.3× per doubling of
  atoms past 1k atoms, on CPU and GPU.
- **Accuracy:** periodic energies and forces agree with direct Ewald, or a
  converged large-cutoff reference, to a stated tolerance.
- **Stability:** NVE drift on a periodic box is comparable to today's cluster
  runs.
- **Compiles:** `compile_energy` still compiles to one graph per topology, with
  no recompiles across neighbour-list rebuilds of the same capacity.
