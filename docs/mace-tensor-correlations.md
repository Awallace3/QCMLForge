# Two-centre tensor correlations

`MACE-AP3D3-H3L3T` is an opt-in short-range pair architecture. It augments
H3L3's axial descriptors with an invariant inner product of the two atoms'
projected l=3 tensors. It does not change the frozen MACE featurizer, the
atomic-property provider, classical energies, or residual cutoff.

Let `A` and `B` be the edge tensors after the same bias-free channel
projection used by H3L3. Both have shape `[edge, 24, 7]`. The additional
feature is `sum(A * B, dim=-1)`, with shape `[edge, 24]`. It is appended
identically to the AB and BA pair vectors. The input width becomes 150
instead of 126. There is no new tensor projection: only the readouts acquire
additional input weights.

An axial descriptor `A · Y_l(u)` measures the angular feature along the
interatomic axis. It is unchanged by rotating that atom's environment around
the axis. `A · B` can distinguish those relative rotations while remaining
invariant to a *joint* rotation or reflection. Other atom pairs can already
encode some of this information through distances; this extension is not a
claim of a universal torsional blind spot or improved prediction accuracy.

## Tensor feature API

```python
from apnet_pt.mace.pair import pair_tensor_invariants

# Same orthonormal irrep basis, degree, parity, and channel projection.
correlations = pair_tensor_invariants(edge_tensors_a, edge_tensors_b)
```

The API returns raw inner products, preserving magnitude. It supports empty
edge batches and autograd, and does not import MACE or e3nn. It does not infer
irrep metadata from tensors: callers must provide matched representations.
Do not feed flattened tensors to a scalar MLP before this contraction, mix
angular components with a generic linear layer, or combine tensors in
different coordinate conventions.

The H3L3T route obtains both sides from the same pinned PolarMACE l=3 block
and shared projection. Its scalar contractions are independent of the
Cartesian-axis permutation; its existing axial descriptors still use the
explicit MACE `(y, z, x)` convention.

## Compatibility and limits

- Existing routes and checkpoint identities remain unchanged.
- H3L3T has its own architecture identity and cannot silently load an H3L3
  state dictionary, or mount an axial-only pair core.
- The source CLI registers `--train_apnet MACE-AP3D3-H3L3T` for its existing
  smoke-data lifecycle. Production dataset orchestration is separate, in the
  paired experiment repository's `scripts/slurm/benchmark.py` and
  `qcmlforge_exp.mace_benchmark`, where the route is named `H3L3T`.
  This is not an implicit warm-start conversion of an existing trained model.
- This route is currently energy-only: the existing live featurizer builds
  graphs through NumPy and detaches its output features, as do caches.
  Pair-coordinate gradients alone are therefore not complete molecular forces.
  Force use requires a separate differentiable-encoder implementation.
- Transferring the features to an AP3D3-FF or CLIFF2 correction requires the
  exact valid frozen baseline and matching residual targets. The feature API
  does not implement baseline transplantation or an exchange multiplier.
- The extension must be evaluated with component-wise errors, not only a
  total error that may improve through cancellation.
