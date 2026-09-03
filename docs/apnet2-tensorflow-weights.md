# Running APNet2 with the original TensorFlow weights

`models/ap2_tf_paper/` holds the five-member AP-Net2 ensemble published with
[`zachglick/apnet`](https://github.com/zachglick/apnet), converted from
TensorFlow SavedModel format to PyTorch checkpoints. Loading them reproduces the
original model's predictions to float32 accumulation noise, so results obtained
this way are numerically the paper's model rather than a retrained
approximation.

Use these checkpoints when you need the published numbers. Use the checkpoints
in `models/` (or your own training run) when you want QCMLForge's own models.

"The original model's predictions" means the TensorFlow forward pass, recorded
and compared dimer by dimer. Against the paper's *reported* MAEs the converted
ensemble matches exchange, induction and dispersion to 2-6e-4 kcal/mol but sits
+0.134 kcal/mol high on electrostatics; see
[Where the converted weights stand against the paper](specs/apnet2-tensorflow-parity.md#where-the-converted-weights-stand-against-the-paper).

## Selecting them by name: `weights="ap2_tf_paper"`

The same checkpoints are published under `ap2_tf_paper/` in the
[`awallace3/qcmlforge`](https://huggingface.co/awallace3/qcmlforge) Hugging Face
repository, so they can be selected by name without cloning this repository or
spelling out paths:

```python
from apnet_pt.pretrained_models import apnet2_model_predict

# Ensemble-averaged prediction from the five published models.
pred = apnet2_model_predict(dimers, weights="ap2_tf_paper")
```

`weights` defaults to `"qcmlforge"`, the ensemble trained by this project, so
existing calls are unaffected. The same keyword selects a single member on
either model class:

```python
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

pair_model = APNet2Model()
pair_model.set_pretrained_model(model_id=0, weights="ap2_tf_paper")

atom_model = AtomModel()
atom_model.set_pretrained_model(model_id=0, weights="ap2_tf_paper")
```

`APNet2Model.set_pretrained_model(model_id=...)` resolves the pair checkpoint
*and* its matching atom checkpoint together, which is what these weights
require: the paper checkpoints are v1 files with no embedded atom submodel, and
each pair model was trained against its own `atom{i}`. It also adopts
`quadrupole_scale = 1.5` from the config, so the named route cannot silently
mis-scale electrostatics.

Downloads obey `QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED` like every other pretrained
artifact. `weights="ap2_tf_paper"` is rejected with a `ValueError` by the fused
routes (`ap2_fused=True`); see [Loading pitfalls](#loading-pitfalls). The
registry of named sets lives in `apnet_pt.hf_pretrained.APNET2_WEIGHT_SETS`
(`apnet2_weight_sets()` lists them), and
`scripts/ap2_tf/upload_paper_models_to_hf.py` is what published them.

## Usage with explicit paths

```python
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

model = APNet2Model(
    pre_trained_model_path="models/ap2_tf_paper/pair_models/pair0.pt",
    atom_model_pre_trained_path="models/ap2_tf_paper/atom_models/atom0.pt",
)
model.model.eval()

# (N, 4) array of [electrostatics, exchange, induction, dispersion] in kcal/mol
components = model.predict_qcel_mols(dimers, batch_size=8)
```

The atom model can be used on its own for multipoles:

```python
atom_model = AtomModel()
atom_model.set_pretrained_model(model_path="models/ap2_tf_paper/atom_models/atom0.pt")
charges, dipoles, quadrupoles, hlist = atom_model.predict_qcel_mols([monomer])[0]
```

The original paper reports ensemble predictions. `apnet2_model_predict(...,
weights="ap2_tf_paper")` does that averaging; by hand, average the five
`pair{i}` models, each paired with its own `atom{i}` — the pair models were trained
against their matching atom model, and the SavedModels embed exactly those
atom-model weights (verified byte-identical).

Both `pre_trained_model_path` and `set_pretrained_model` work, and both now
adopt `quadrupole_scale` from the checkpoint config; see
[Loading pitfalls](#loading-pitfalls).

## Verified parity

`tests/test_ap2_tf_parity.py` compares every converted model against recorded
TensorFlow output for 24 dimers drawn from the processed test split (elements
H, C, N, O, F, P, S, including charged monomers). Atom deviations are on the
10-element multipole vector; pair deviations are in kcal/mol.

| model | atom max abs Δ (A) | atom max abs Δ (B) | pair component max abs Δ | pair total max abs Δ | total MAE (PT) | total MAE (TF) |
|---|---|---|---|---|---|---|
| 0 | 5.96e-07 | 1.22e-06 | 6.48e-05 | 6.40e-05 | 0.2520 | 0.2520 |
| 1 | 5.96e-07 | 3.22e-06 | 9.16e-05 | 7.77e-05 | 0.3366 | 0.3366 |
| 2 | 5.96e-07 | 1.88e-06 | 8.33e-05 | 8.19e-05 | 0.2563 | 0.2563 |
| 3 | 7.75e-07 | 1.05e-06 | 1.21e-04 | 1.21e-04 | 0.2487 | 0.2487 |
| 4 | 5.96e-07 | 1.25e-06 | 5.34e-05 | 7.72e-05 | 0.1834 | 0.1834 |

Every atom-model weight transfers bit-exactly (135 of 135 tensors, verified
against the SavedModel variable bytes), so the residual 1e-6 on multipoles is
purely reduction-order difference in float32. The 1e-4 kcal/mol on energies is
the same effect after propagation through the pair network. Nothing here is an
approximation with a tunable knob: there is no remaining known systematic
difference between the two forward passes.

The MAE columns are over the same 24 dimers and agree to four decimals, which is
the check that matters for users — the conversion does not shift accuracy.

## Loading pitfalls

**`quadrupole_scale` must be 1.5.** TensorFlow's `KerasPairModel.mtp_elst`
multiplies both quadrupole tensors by `3/2` before contracting them with the
`T2` interaction tensor. QCMLForge exposes that factor as `quadrupole_scale`
and defaults it to `1.0` to preserve historical behavior. The converted pair
checkpoints therefore carry `quadrupole_scale: 1.5` in their config.

This is a forward-pass constant, not a parameter, so it does not live in the
state dict. A loader that only calls `load_state_dict` will load the weights
successfully and still predict the wrong electrostatics, with no error and no
warning. On the 24 parity dimers, dropping 1.5 to 1.0 moves electrostatics by
0.086 kcal/mol on average and up to 0.498 kcal/mol, and moves exchange,
induction and dispersion by exactly zero. Against labels the damage is smaller
than that spread suggests — some of it cancels — but still 0.012 kcal/mol of
Elst MAE on the paper's validation split.
`APNet2Model.set_pretrained_model` now reads it out of the checkpoint config.
If you write your own loader, do the same.

**`APNet2_AM_MPNN` (`apnet2_fused.py`) cannot load these checkpoints.** The
fused model combines the atom and pair networks into a single state dict, while
the TensorFlow pair SavedModel and the converted 83-tensor checkpoint keep them
separate. `apnet2_fused.py::set_pretrained_model` also does not adopt
`quadrupole_scale` from a checkpoint config, so a fused checkpoint trained with
`--quadrupole-scale 1.5` will silently load at `1.0`. Use the unfused
`APNet2Model` for the converted weights; the ensemble routes raise a
`ValueError` rather than mis-load when `weights="ap2_tf_paper"` is combined with
`ap2_fused=True`.

**Predictions were verified on CPU.** float32 reduction order differs between
CPU and GPU, so GPU predictions will differ from the table above at roughly the
same 1e-4 kcal/mol magnitude. That is expected and is not a loading error.

## Earlier converted checkpoints were broken

If you used `models/ap2_tf_paper/pair_models/*.pt` from before this revision, discard
those results. Only 19 of the 83 pair tensors carried genuine TensorFlow
weights. The rest — feed-forward sublayers 2, 4 and 6 of every stack, the
element embedding table, and `distance_layer_im.frequencies` — retained their
PyTorch initialization; their values match Kaiming-uniform and `pi * (1..8)`
init signatures exactly and match nothing in any of the 20 published
SavedModels. Worse, `readout_layer_indu.0.*` was left as an
`UninitializedParameter`, so `nn.LazyLinear` materialized the induction readout
from the ambient torch seed on first forward: the induction component was a
different random function on every process.

`load_state_dict` accepted all of this, and the existing smoke tests passed,
because the energies remained finite and kept the right signs. Measured against
labels on the 24-dimer set, total MAE was 48.1 / 12.3 / 275.5 / 36.7 / 20.2
kcal/mol for models 0–4, against TensorFlow's 0.25 / 0.34 / 0.26 / 0.25 / 0.18.

**This, not the `sparse`/`master` question below, explains the accuracy
discrepancy previously seen against the paper.** The atom models were always
correct; the pair models were never usable.

`tests/test_ap2_tf_parity.py` now asserts that all 83 tensors are present,
finite, and non-lazy, so this specific failure cannot recur silently.

## Which upstream branch these come from

The conversion source is branch `sparse` at commit
`f093e00bf64190ac30a7706d2a90e66871347b76` (tag `v0.1.0`), recorded in each
checkpoint's `tf_provenance` field along with the SavedModel's
`saved_model.pb` SHA-256.

`master` in the same repository is **AP-Net v1**, not an older AP-Net2. It uses
ACSF/APSF symmetry-function descriptors instead of message passing, runs in
float64 (`set_floatx('float64')`), and reads a different dataset format
(`hive*.h5`, `hfadz*.hdf5`). Its weights have no counterpart in the QCMLForge
APNet2 architecture and cannot be converted; the difference is architectural,
not a version skew. So `master` is not a candidate explanation for any accuracy
gap in APNet2 results.

Within `sparse` there are two SavedModel vintages, and this one does matter:
`atom{i}`/`pair{i}` use a 119-row element embedding table, while
`atom{i}_old`/`pair{i}_old` use 36 rows. The checkpoints in `models/ap2_tf_paper/`
descend from the **new** vintage. `scripts/ap2_tf/tf_reference_predictions.py`
takes a `--vintage` flag if you need to compare against the old one.

## Regenerating the conversion

Ordinary use of `models/ap2_tf_paper/**` needs nothing but QCMLForge. Regenerating
the checkpoints or the test fixture needs the legacy TensorFlow environment,
because TF 2.3 is the last release that can read these SavedModels and its
wheels stop at python 3.8:

```bash
# Builds a pinned python 3.8 / TF 2.3.4 prefix and loads all 20 SavedModels.
scripts/ap2_tf/make_legacy_tf_env.sh
```

`scripts/ap2_tf/legacy-tf-env.pip-freeze.txt` records the environment that
actually produced the current artifacts. Then, in that environment:

```bash
# 1. Export SavedModel variables to npz + manifest (legacy env).
python scripts/ap2_tf/export_tf_savedmodel.py \
    <apnet>/apnet/atom_models/atom0 <apnet>/apnet/pair_models/pair0 \
    --out-dir tf_npz

# 2. Convert to a PyTorch checkpoint (either env; needs torch).
python scripts/ap2_tf/convert_tf_to_pt.py --kind atom --npz tf_npz/atom0.npz \
    --out models/ap2_tf_paper/atom_models/atom0.pt --overwrite
python scripts/ap2_tf/convert_tf_to_pt.py --kind pair --npz tf_npz/pair0.npz \
    --out models/ap2_tf_paper/pair_models/pair0.pt --overwrite
```

The converter walks the exported variables positionally and asserts on every
TensorFlow variable name it consumes, so a shape or ordering change in the
source aborts rather than producing a partly-populated checkpoint. It also
asserts that it consumed exactly 135 variables for an atom model and 218 for a
pair model (the pair SavedModel embeds the frozen atom model in its first 135).

Regenerating the test fixture is a two-step, two-environment process, because
the dimers come from processed PyTorch shards and the reference numbers come
from TensorFlow:

```bash
# PyTorch env: select dimers from the processed shards.
python scripts/ap2_tf/make_parity_dimers.py \
    --processed-dir <data_dir>/processed \
    --prefix dimer_ap2_fused_test_spec_2_ --samples 24 \
    --out-npz tests/dataset_data/ap2_tf_parity/parity_dimers.npz \
    --out-manifest tests/dataset_data/ap2_tf_parity/parity_dimers.manifest.json

# Legacy env: run TensorFlow on exactly those dimers.
python scripts/ap2_tf/tf_reference_predictions.py \
    --dimers-npz tests/dataset_data/ap2_tf_parity/parity_dimers.npz \
    --out-npz tests/dataset_data/ap2_tf_parity/tf_reference.npz \
    --out-manifest tests/dataset_data/ap2_tf_parity/tf_reference.manifest.json
```

`scripts/ap2_tf/parity_common.py` is shared by both steps and depends only on
numpy and qcelemental so it can be imported by the python 3.8 interpreter. It
builds molecules with `validate=False` and `fix_com`/`fix_orientation` so
qcelemental's molparse cannot re-center or reorient one environment's geometry
relative to the other's, and asserts that the Angstrom coordinates survive the
round trip float32-exactly. Without that, a comparison at the 1e-6 level is
measuring molparse, not the models.

## Related

- `tests/test_ap2_tf_paper_route.py` — covers the `weights="ap2_tf_paper"`
  route: the registry, the rejected argument combinations, and numeric
  agreement between the named route and explicit paths.
- [APNet2 TensorFlow/PyTorch parity controls](specs/apnet2-tensorflow-parity.md)
  — the training-time differences between the two implementations, and why they
  are hard to resolve experimentally.
