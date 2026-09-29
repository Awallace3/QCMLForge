# Frozen MACE features for frame-free exponent-anisotropic exchange

`apnet_pt.mace.MACEMASTIFFExchange` is an opt-in exchange-only model, separate
from AP3 residual heads and `MACEExchangeCorrection`. It replaces a learned
atomic parameter producer with frozen pretrained MACE descriptors; it does
not add an unconstrained energy correction to a parent.

## Kernel and features

For each atom, invariant descriptors feed an MLP predicting corrections to
explicit element-baseline log A and log B. A is in sqrt(kcal/mol), B in inverse
angstrom. Positive A/B are obtained by exponentiation. Unsupported elements
are errors, not silent baseline substitutions.

For each natural-parity degree l = 1, 2, 3, a bias-free linear map mixes only
the MACE channels, identically for every m component:

```text
raw_c_l = channel_projection_l(MACE_l)
c_l = rho * raw_c_l / sqrt(1 + ||raw_c_l||²)
s_i(u) = sum_l c_i,l · C_l(u)
x = r sqrt(B_i B_j exp(s_i(u)) exp(s_j(-u)))
E_exch,ij = A_i A_j (1 + x + x²/3) exp(-x)
```

Default rho is 0.4 per degree. `C_l` has Racah normalization, norm one on the
unit sphere; e3nn component-normalized harmonics are divided by sqrt(2l+1).
Both the MACE axis permutation and odd-degree endpoint parity are explicit.
There are no RDKit frames, chemical masks or orientation-dependent biases.
The angular factor exp(s) has lower bound exp(-3*rho). Exchange is positive
mathematically, although numerical underflow can produce zero at large
separations. Nonfinite parameters/energies fail rather than silently passing
NaNs into optimization.

Anisotropy changes the entire overlap argument x, including the polynomial.
This is **not** the older linear anisotropic-prefactor MASTIFF kernel.
`apnet_pt.mastiff_exponent.slater_exponent_exchange` exposes the pure tensor
kernel independently of the feature producer.

`anisotropy=False` gives an isotropic control with the same scalar-head
architecture and no unused angular parameters. Parameter counts differ;
report them rather than claiming exact capacity matching.

## Energy interface

```python
from apnet_pt.mace import MACEMASTIFFExchange

head = MACEMASTIFFExchange(
    mace_feature_dim=features_a.invariant.shape[1],
    feature_schema=features_a.feature_schema,
    mace_checkpoint_sha256=verified_artifact_sha256,
    element_baselines=locked_element_baselines,  # Z -> (A, B)
).to(device=positions_a.device, dtype=positions_a.dtype)

# edge_index: int64 [2, n_pair], indices into separate A and B atom arrays.
pair_exchange = head.pair_energies(
    features_a, features_b, positions_a, positions_b, edge_index
)
dimer_exchange = head(
    features_a, features_b, positions_a, positions_b, edge_index
)
```

Coordinates are angstrom, outputs kcal/mol. `forward` returns `[n_dimer]`;
`pair_energies` returns `[n_pair]`. Atom-to-dimer assignments come from the
feature containers. Cross-dimer edges are rejected. The caller must preserve
feature/coordinate atom ordering and supply every desired pair exactly once.
There is **no intermonomer cutoff**; use all A–B pairs for full exchange,
not a silently truncated AP3 short-range graph.

The MACE featurizer remains external and frozen. The head detaches its inputs
even if callers accidentally enable their gradients. This is energy-only:
autograd through the supplied coordinates omits feature-response derivatives
and must not be advertised as complete molecular forces.

`atom_quantities(features)` returns `(A, B, (c_l1, c_l2, c_l3))`. All parameters
are materialized at construction. Zero `readout_init_scale` starts exactly at
the supplied isotropic element baselines; the default 0.01 starts near them.
Neither setting guarantees equality with arbitrary CLIFF2/AP3D3-FF exchange.

## Checkpoints and integration boundary

Save `{"config": head.get_config(), "state": head.state_dict()}` using
`torch.save`; load with `torch.load(..., weights_only=True)`, construct from
the saved config, move to the saved dtype/device, and load with **strict=True**.
The state contains only heads/baselines, never a MACE backbone. Extra state
binds the architecture version, feature schema, external artifact SHA-256,
element baselines, rho, isotropic/angular choice and initialization setting.
Missing or mismatched extra state is an error under strict loading. Do not
strip metadata or use non-strict loading to bypass that contract.

This is not a `MACEAP3D3` v3 checkpoint and is not yet a registered
`train_models.py --train_apnet` route. Experiment trainers may consume the
explicit interface; a general CLI/training harness is separate integration.

To use it with a parent, explicitly replace the desired exchange contribution
and preserve the other components. Distinguish replacement of classical CLIFF
exchange from replacement of classical-plus-neural AP3D3-FF exchange; do not
double-count or silently remove a residual. Full parent loader and component
reproduction, row alignment and training/evaluation exposure checks remain
necessary before total-energy comparisons.
