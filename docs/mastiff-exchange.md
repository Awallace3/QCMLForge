# Explicit MASTIFF exchange

`apnet_pt.mastiff` is an opt-in differentiable exchange kernel, not a replacement
for an existing CLIFF or AP3 route. It implements Eqs. (2–3) of
[J. Phys. Chem. A 2023, 127, 1736–1749](https://doi.org/10.1021/acs.jpca.2c07244)
and the exchange expression in the
[OpenMM-CANplugin benzene example](https://github.com/jrschmidt2/OpenMM-CANplugin/blob/de25a9847ec54335b03651bb48ebfa8953376b7e/examples/MASTIFF_benzene_runfiles/benzene_2b_3b_lrc_noljpme_07162021_no15.xml).
The equations were implemented independently; no plugin source is vendored.

## Exact functional form

\[
E_{ij}= A_{i,\mathrm{iso}} A_{j,\mathrm{iso}}
 (1+\sum a_{i,lm} C_{lm}(\hat r_i))
 (1+\sum a_{j,lm} C_{lm}(\hat r_j))
 [1+x+x^2/3] e^{-x},\qquad x=\sqrt{B_i B_j}\,r_{ij}.
\]

Here `r_i = F_i.T @ (R_j-R_i)` and
`r_j = F_j.T @ (R_i-R_j)`; frame **columns** are local x,y,z axes in the
laboratory frame. `B_i` is positive and isotropic. There is no anisotropic
exponent, bounded exponential angular factor, clamp, or implicit conversion.
Coordinate and exponent units must be reciprocal; output units are the product
of the A units. CAN's example uses nm, nm^-1, and sqrt(kJ/mol). CLIFF amplitudes
in sqrt(hartree) with B=1/sigma and bohr distances require an explicit
hartree-to-kcal/mol conversion after evaluation. This module does not impose
CLIFF's width floors or ceilings.

The linear angular factor can be negative; positivity must NOT be manufactured
by changing the paper's equation. Indeed the supplied benzene H coefficients
produce a negative factor for some directions. Any future positivity constraint
would be a separately identified fitting policy, not this kernel's definition.

## Real harmonic convention

All eight real channels through l=2 are supported, ordered:

`10, 11c, 11s, 20, 21c, 21s, 22c, 22s`.

These are Racah-normalized `C_lm = sqrt(4*pi/(2*l+1)) Y_lm`, with
positive Cartesian real phases. For unit local coordinates (x,y,z):

- l=1: `z, x, y`;
- l=2: `(3*z*z-1)/2, sqrt(3)*x*z, sqrt(3)*y*z,
  sqrt(3)/2*(x*x-y*y), sqrt(3)*x*y`.

CAN's CUDA azimuth is unsigned/folded; its cosine-only benzene expansion is
unaffected, but it cannot serve as a reference for general sine channels.
This module intentionally retains signed local Cartesian coordinates and thus
true sine harmonics. Frames require unwrapped monomer coordinates (no implicit
periodic boundary handling).

These polynomials **are** exact real spherical harmonics, not electrostatic
multipole projections. `RacahHarmonics(backend="cuequivariance")` instead uses
NVIDIA's library and explicitly maps its polar-y, component-normalized basis
onto the same convention (input cycle y,z,x, channel reorder, divide by
sqrt(2*l+1)). Install `pip install -e '.[mastiff]'`. Set `method="naive"` for CPU;
leave method unset for the CUDA backend. The default `torch` backend is a small
CPU/GPU reference and requires no optional dependency. Both avoid angular
`acos`/`atan2` singularities at poles. Higher l is not yet supported.

## Usage

```python
import torch
from apnet_pt.mastiff import MastiffExchange, local_frames

# Arrays are aligned per intermolecular edge; gather atomwise quantities first.
# ref_z and ref_x are reference-neighbor minus central-atom vectors.
Fi = local_frames(ref_z_i, ref_x_i, kind="z-then-x")
Fj = local_frames(ref_z_j, ref_x_j, kind="z-then-x")
exchange = MastiffExchange(backend="cuequivariance", method="naive")
E_edges = exchange(Rj - Ri, Ai, Aj, Bi, Bj, coeff_i, coeff_j, Fi, Fj)
# Sum by dimer using the existing scatter_sum_compile utility.
```

`A` and `B` have shape `(edges,)`, coefficients `(edges,8)`, frames `(edges,3,3)`.
Zero coefficients reproduce the isotropic Slater form. Per-type learnable
coefficients or an invariant MPNN's eight-channel readout may provide the
coefficients **only after** a symmetry-consistent body frame has been assigned.
All amplitudes, exponents, coefficients, and frame-defining coordinates retain
autograd paths. Positive parameter heads for A and B should be explicit at the
model boundary. This initial module is not yet wired into the CLIFF trainer,
checkpoint schema, or dataset pipeline.

## Frames: what generalizes and what does not

The five CAN-style geometric constructions are available: `z-then-x`,
`bisector`, `z-bisect`, `threefold`, and `z-only`. Supply relative reference
vectors from a fixed monomer topology. For `bisector`, this API's second
vector is the second bisected bond (CAN's AtomY); X is normal to their plane.
CAN averages raw displacements, not normalized bonds, so unequal bond lengths
matter. This implementation preserves those sums. Validate each conformer:
zero bonds, collinearity and cancelling bisectors fail rather than silently
creating a laboratory-fixed non-axial frame. `validate=False` is a compiled
fast path **only for already validated nondegenerate inputs**.

Both `z-only` and CAN-style `threefold` frames have an arbitrary transverse
gauge; restrict them to `axial` coefficients. A threefold frame with nonzero
m=3 would require a chemically defined transverse axis and is not supported.
The implementation replaces CAN's length-dependent lab-axis threshold with a
nonsingular gauge; no non-axial physical equivalence is claimed. The exchange constructor supports `general`, `c2v` (10,20,22c),
`axial` (10,20), and `isotropic` masks independently at the two ends. For mixed
site symmetries, group edges by endpoint symmetry or supply correctly masked
coefficients using general mode. Masking is part of the caller's physical
contract; do not use general coefficients with a z-only or threefold frame.

Automatic assignment cannot mean sorting neighbor atom indices or choosing a
PCA eigenvector sign. Those choices can change under permutation or become
singular at symmetric configurations. Degree alone also does not establish
local chemical symmetry (e.g. every terminal atom is not cylindrically symmetric).

Recommended first integration: topology/chemical-environment templates with
explicit frame type, reference selection, and allowed harmonic channels. Start
with benzene's published C/H conventions and cover other environments only
with validated templates. Unsupported or degenerate environments must be
reported; do not silently classify them as isotropic. Equivalent reference
choices must either leave the allowed basis invariant or be symmetrized at the
**atomic angular-factor** level before multiplying the two ends. Swapping the
two ring-carbon x references for benzene flips x and y, but leaves C10, C20,
C22c unchanged; it does not preserve a general eight-channel expansion.

A more general alternative is a frame-free equivariant coefficient field:
learn independent irreducible tensors `t_i^(l)` from monomer geometry and
contract with `Y_l(rhat_ij)` in a matched normalization. This retains explicit
SH dependence while eliminating axis selection and electrostatic-multipole
shape locking. It requires an equivariant producer, not eight scalar MPNN
outputs interpreted in the laboratory frame. It is a MASTIFF-inspired extension,
not a reproduction of the published atom-type/body-frame parametrization.

## Validation and limits

The focused suite checks analytic harmonics, cuEquivariance values/gradients,
CAN benzene energy-expression parity, isotropic limit, odd-l endpoint signs,
rotations, frame construction and autograd. No OpenMM CUDA runtime comparison,
trained force-field accuracy claim, or replacement-checkpoint claim is implied.
No existing checkpoint needs conversion because no existing route changes.
