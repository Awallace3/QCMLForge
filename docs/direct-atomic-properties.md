# Direct energy-trained atomic properties

`apnet_pt.AtomModels.direct_atomic_properties.DirectAtomicProperties` is a
checkpoint-independent `nn.Module`. It takes `x`, `R`, `edge_index`,
`molecule_ind` and `total_charge` on a monomer graph, and returns absolute
`q [N]`, `mu [N,3]`, `quad [N,3,3]`, and `hfvr [N]`.

- Coordinates are Angstrom. Moments are in atomic units, with symmetric,
  traceless quadrupoles in the package's convention.
- Both Q and Q/N condition the message-passing network. Charges are projected
  to the known monomer total, not predicted independently of charge state.
- Vector/tensor heads contract directed intramonomer neighbor unit vectors;
  no lab-frame vector is learned. Monatomic dipoles/quadrupoles are zero.
- HFVR is a bounded positive sigmoid (default bounds 0.001--2.0, reference 1.0).
  Effective polarizability is `alpha_free(Z) * hfvr ** (4/3)`.
- Small nonzero readout initialization avoids starting every neutral field at
  zero. These properties are not trained to MBIS or Hirshfeld labels.
- Validate graphs at the data boundary: finite coordinates/charges, valid
  elements, no self/coincident/cross-monomer edges, positive atom counts and
  matching network/dataset cutoffs. Forward avoids atom-wise Python loops.

`explicit_thole_induction` in `AtomPairwiseModels/mtp_mtp.py` takes explicit
side-local coordinates, q, mu, alpha, AB/AA/BB edges and dimer indices. It
uses fixed global direct and mutual Thole constants (default 0.38539), the
existing Rackers damping convention, and self-consistent mutual relay.
Only intermolecular permanent fields drive the response. Quadrupoles do not
enter induction, matching the existing solver. Energies are kcal/mol.

No widths, overlap parameters, pretrained predictor or post-solve scaling
are needed. The default maximum-component convergence criterion prevents
batch-size dilution of a difficult dimer's residual. Nonfinite/nonconverged responses raise; bounds and positivity
alone are not a proof of stability. Energy and property gradients remain
connected through the iterative solve.

Energy-only fitting does not uniquely identify atomic moments or volumes.
Call the outputs effective properties. Check out-of-domain transferability,
long-range behavior, property ranges and polarization stability separately.

Synthetic verification (set `PYTHONPATH` to this worktree's `src`):

```sh
python -m pytest tests/test_direct_atomic_properties.py tests/test_direct_induction.py
```
