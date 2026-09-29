# Parent-preserving MACE exchange correction

`apnet_pt.mace.MACEExchangeCorrection` is an opt-in **energy-only** module.
It consumes frozen isolated-monomer MACE features and already verified parent
component energies. It does not load a parent checkpoint, replace the atomic
forward, or register a new production training CLI route.

```python
from apnet_pt.mace import MACEExchangeCorrection

correction = MACEExchangeCorrection(
    mace_feature_dim=features_a.invariant.shape[1],
    mace_equivariant_dim=features_a.equivariant_degree(3).shape[1],
    feature_schema=features_a.feature_schema,
    parent_contract_sha256=verified_parent_manifest_sha256,
    mode="axial-tensor",  # matched orientation-blind control: "scalar"
    tensor_scaling="unit-ball",  # independent alternative: "raw"
)
energies = correction(batch, features_a, features_b, parent_components)
```

Inputs use angstrom coordinates and kcal/mol energies. Parent components must
have shape `[n_dimer, 4]`, ordered `[elst, exch, indu, disp]`. The batch must
provide `ZA`, `ZB`, `RA`, `RB`, `e_ABsr_source`, `e_ABsr_target`, and `dimer_ind`.
Feature monomer batch indices must match each edge's parent-dimer index.
Move the model and all floating inputs to the same device/dtype before use.

## Contract

- The last scalar readout is zero-initialized: the first prediction exactly
  reproduces the parent.
- Only exchange changes during learning. The other three components are copied,
  not re-evaluated or refit. Parent and input MACE tensors are detached; no
  gradients flow into them.
- A shared readout is averaged over AB and BA. Natural-parity spherical harmonics
  use MACE's `(x,y,z) -> (y,z,x)` convention, making the descriptor invariant
  under joint rotations/reflections and the correction monomer-swap symmetric.
- A cosine envelope makes the correction zero beyond the configured cutoff
  and gives zero radial slope at the boundary. Empty edge lists return the
  parent unchanged.
- All parameters exist before the first forward; there are no lazy layers.
- `get_config()` reconstructs the module. Strict `state_dict` loading checks
  architecture, parent-contract hash, exact feature schema, descriptor mode,
  scaling, dimensions and cutoff.

Use `load_state_dict(..., strict=True)` (the PyTorch default). Permissive loading
with stripped `_extra_state` forfeits these identity checks and is not a
supported checkpoint-loading contract.

## Controlled descriptors

For each matched irrep channel, `scalar` supplies the two tensor norms and their
product. `axial-tensor` supplies each tensor's contraction onto its directed
interatomic axis and their common-frame inner product. The same learnable
scalar channel projection acts after contraction in both modes. Both expose
the same number of readout inputs and trainable parameters; this is a
parameter-count control, not an assertion of identical input distributions.

`unit-ball` first maps each tensor channel `F` to `F/sqrt(1+||F||²)`.
This preserves equivariance, bounds magnitudes without a zero-norm division,
and retains amplitude information. It is not a learned overlap law.
`raw` retains the original amplitudes. Record scaling explicitly in comparisons.

## Parent provenance is a caller responsibility

The parent contract hash is an identity guard, **not authentication of input
tensor contents**. The caller must bind row/geometry identities, component
order/units, parent checkpoint hashes, producer source/experiment revisions,
forward settings and payload hashes. Reject stale or reordered exports.
Audit the parent's own training membership against any purported held-out
evaluation set; freezing a previously trained model does not remove leakage.

This separation permits a compatible parent producer to export predictions
without importing its model into an incompatible source branch. It does not
make corrected weights safe under an old atomic forward. For residual parents,
the contract must include the full classical-plus-neural composition, not
merely the residual checkpoint hash.

There is no positivity constraint on the learned delta or corrected exchange.
Include physical stress tests before extrapolation or deployment. Coordinate
gradients through this small head omit derivatives through the detached MACE
features and parent energies: they are **not molecular forces**.
