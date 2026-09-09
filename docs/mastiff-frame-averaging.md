# Symmetry-equivalent MASTIFF frame averaging

`apnet_pt.mastiff_frames.AveragedFrameHarmonics` evaluates explicit local-frame
plans without depending on electrostatic multipole directions. It is a reusable
geometry operation, not a chemical perception engine or a pretrained model.

Supply monomer positions `(N,3)`, outgoing partner directions `(N,M,3)`, and:

- full-frame rows `(center, z_reference, x_reference)`;
- axial rows `(center, z_reference)`.

Multiple rows for one center are equally averaged **at the atomic harmonic
feature level**, before contracting with coefficients and multiplying atomic
factors. Disjoint full/axial plans are enforced when validation is enabled.
Atoms absent from both plans are explicitly isotropic; callers must record that
assignment rather than treating missing typing as an automatic isotropic fallback.
Indices must be local to one monomer. The geometry path is vectorized and
coordinate-differentiable. `validate=False` removes boundary validation for
prevalidated inputs; it does not regularize singular frames.

## Symmetry and expressivity

The full-frame representation retains real Racah channels
`10, 11c, 20, 21c, 22c`. It deliberately suppresses sine channels
`11s, 21s, 22s`. A frame built from two polar reference vectors has an axial
cross-product Y axis; under a laboratory reflection, the local Y coordinate
changes sign. Masking sine channels ensures reflection-invariant scalar
contractions with graph-based scalar coefficients. This is a **restricted
angular representation**, not proof that arbitrary atomic environments have
mirror symmetry. More expressive chirality-sensitive representations require
appropriately transforming coefficients and additional conventions.

Axial sites use only `10,20`, evaluated directly from the outgoing direction's
projection on the Z bond. They do not require a laboratory-fixed transverse
axis. Symmetry-equivalent frame averaging can cancel entire harmonic sectors;
for example a tetrahedrally symmetric environment has no free l=1,2 anisotropy.
Increasing parameter count cannot recover angular channels absent from the
chosen basis.

A chemical preprocessor must assign reference roles from graph invariants or
explicit templates. Breaking equivalent-neighbor ties by atom index is invalid.
Averaging all tied assignments is safe even when equivalent neighbors differ
geometrically in a particular conformer. Reject collinear full-frame references;
never choose a different reference as a geometry-dependent fallback without
separately addressing continuity.

Tests cover proper rotations, reflections, atom permutation with remapped plans,
reference-row reordering, coordinate gradients, invalid plans, isotropic sites,
and agreement between PyTorch and cuEquivariance harmonics. Chemical coverage
and training performance must be established separately for each preprocessing
and atom-typing rule.
