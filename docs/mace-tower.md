# MACE-fused AP-Net3-D3

`apnet_pt.mace` lets an AP-Net3-D3 pair model read its atomic representation
from a MACE foundation backbone instead of from AP-Net's own atomic MPNN. The
backbone stays external: it is loaded from a local file by digest, frozen, and
named — never copied — by the checkpoints this package writes.

The whole subpackage is optional. `import apnet_pt` does not import MACE, and
`apnet_pt.mace` itself only exposes the schema dataclasses eagerly; every
adapter is resolved lazily through `__getattr__`, so a base install with no
`mace-torch` keeps working.

```bash
pip install -e ".[mace]"
```

## The external artifact is separately licensed

MACE-POLAR-1-S ships under the ACEsuit **Academic Software License**, not
QCMLForge's MIT. It is therefore never vendored, committed, packaged, or
redistributed here. `setup.cfg` excludes `*.model` and `*.ckpt` from package
data so that a local working copy cannot be swept into a wheel, and
`load_verified_polar_mace` refuses any file whose SHA-256 is not
`POLAR_1S_SHA256`. MACE imports happen only *after* that check passes.

Download it yourself from `POLAR_1S_URL`
(`apnet_pt.mace.encoder`), accept the upstream licence, and point the code at
your copy.

## Feature modes

A featurizer reads one of two things out of the backbone. Both widths are
fixed by the artifact and are named as constants so a pair core can be sized
before a featurizer exists:

| `feature_mode` | invariant | equivariant | needs private adapter |
|---|---|---|---|
| `final-layer-scalars` | 512 | 0 | no |
| `all-scalars+norms` | 2560 | 512 channels (`512x0e+512x1o+512x2e+512x3o`, 8192 columns) | yes |

`final-layer-scalars` uses only the public MACE forward. `all-scalars+norms`
additionally needs `PolarMACEPrivateLayerAdapter`, which reaches into the
version-pinned internals of `mace-torch` — hence `SUPPORTED_MACE_VERSION`.
Only `all-scalars+norms` exposes an equivariant block, so only it can feed a
directional route.

## Routes

`PAIR_ROUTE_CONFIGS` in `apnet_pt.mace.pair` maps an architecture id to its
`(pair_mode, feature_mode)`. The architecture id is what a checkpoint records
and what resume compares, so variants that change the readout width or the
physics the head sees get distinct ids rather than constructor flags.

- `hybrid-h1` — MACE scalars into the AP-Net3 atomic slot, public features only.
- `hybrid-h2` — `all-scalars+norms`, AP-Net3's intramonomer update stack bypassed.
- `hybrid-h3`, `hybrid-h3l1`, `hybrid-h3l3` — as `h2`, plus a contraction of the
  backbone's degree-2, degree-1, or degree-3 equivariant block into the pair
  model's directional slot (`DIRECTIONAL_DEGREES`).
- `hybrid-h3l3q` — `hybrid-h3l3` plus six monomer conditioning scalars
  (`MONOMER_CONDITIONING_SCALARS`, both monomers).
- `hybrid-h3l3w112`, `hybrid-h3l3p`, `hybrid-h3l3w112p` — widen the directional
  slot from its canonical 24 to 112, give each of the four readouts its own
  directional projection, or both.
- `direct-polar`, `atomhead` — `h1` pair mode, but a different source for the
  atomic properties the classical terms are built from: `direct-polar` takes
  PolarMACE's own charge/multipole/polarizability outputs, `atomhead` learns
  completion heads on the MACE features, and every `hybrid-*` route above
  keeps AP-Net's pretrained atomic MPNN (`provider_kind` in
  `MACE_AP3D3_ARCHITECTURES`).

Any contraction against MACE's equivariant output must apply
`MACE_E3NN_AXIS_PERMUTATION`, the Cartesian `(x, y, z) -> (y, z, x)` reordering
MACE feeds its own spherical harmonics. Getting it wrong is not an error — the
result is merely not rotationally invariant, and it still trains — so
`tests/test_mace_routes.py` asserts the constant against the MACE function.

## v3 checkpoints

`MACEAP3D3.save_checkpoint_v3` writes the trained parts and *omits* every
tensor under `featurizer.backbone.`, recording instead the artifact's
`sha256`, `model_id`, `model_class`, canonical locator, and licence
acknowledgement. `config["parameter_counts"]["external"]` accounts for the
backbone by count while the file carries none of it.

`load_checkpoint_v3` therefore needs the artifact back:

```py
model = MACEAP3D3.load_checkpoint_v3(
    path,
    mace_artifact_path="MACE-POLAR-1-S.model",
    model_factory=lambda config, backbone: ...,
    backbone_loader=lambda artifact, *, map_location: load_verified_polar_mace(
        artifact, expected_sha256=POLAR_1S_SHA256
    ),
)
```

Reconstruction is validated, not assumed: architecture, pair mode, backbone
class, feature mode, resolved feature schema, dtype policy, atomic property
schema, and the physics hash all have to match what the record claims. A
checkpoint that silently loads against a different backbone would produce
wrong numbers with no error, which is the failure this validation exists to
prevent.

## `graph_longrange` compatibility

`_graph_longrange_compat` patches one missing `dim_size` in
`graph_longrange` 0.4.0's real-space scatter. Without it, a batch whose
*trailing* monomer contributes no edges — a monatomic monomer, whose
duplicates are all masked out of the complete graph — truncates the per-node
energy vector and fails the next scatter. The patch is a no-op for every other
batch and is keyed to the digest of the defective source.

## Tests

The stub harness (`tests/mace_stub_harness.py`) covers routes, checkpoints,
and the compatibility patch with an invented backbone, and needs no artifact.

Two properties cannot be stubbed: the declared feature widths (a stub agreeing
with them proves only that two invented numbers match) and the size floor the
packaging exclusion defends (the stub backbone is 800 kB). Those live in
`tests/test_mace_integration_polar.py`, behind a digest-verified gate:

```bash
# skips cleanly when unset
QCMLFORGE_POLARMACE_ARTIFACT=/path/to/MACE-POLAR-1-S.model pytest -m mace_integration

# fails instead of skipping, for a CI job that is supposed to have the artifact
QCMLFORGE_REQUIRE_POLARMACE=1 pytest -m mace_integration
```

The gate checks the file's size and streamed SHA-256 before any test runs, so
a wrong or truncated copy fails as a wrong copy rather than as a shape error
several layers in.
