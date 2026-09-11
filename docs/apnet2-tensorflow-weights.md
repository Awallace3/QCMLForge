# AP-Net2 paper weights

`models/ap2_tf_paper/` contains the five-member ensemble from
[`zachglick/apnet`](https://github.com/zachglick/apnet), converted to PyTorch.
The source is the new, 119-element-embedding vintage on branch `sparse`,
commit `f093e00bf64190ac30a7706d2a90e66871347b76` (tag `v0.1.0`).
Each checkpoint records the source revision and SavedModel hash in
`tf_provenance`. Upstream `master` is AP-Net v1, not AP-Net2.

## Usage

```python
from apnet_pt.pretrained_models import apnet2_model_predict

# Five-member ensemble; output columns are elst, exch, indu, disp in kcal/mol.
components = apnet2_model_predict(dimers, weights="ap2_tf_paper")
```

The default remains `weights="qcmlforge"`. Named weights download from
[`awallace3/qcmlforge`](https://huggingface.co/awallace3/qcmlforge), respecting
`QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED`. Select a single member with:

```python
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

model = APNet2Model()
model.set_pretrained_model(model_id=0, weights="ap2_tf_paper")
```

`AtomModel.set_pretrained_model` accepts the same keyword for multipole-only
predictions. For local checkpoints:

```python
model = APNet2Model(
    pre_trained_model_path="models/ap2_tf_paper/pair_models/pair0.pt",
    atom_model_pre_trained_path="models/ap2_tf_paper/atom_models/atom0.pt",
)
components = model.predict_qcel_mols(dimers, batch_size=8)
```

## Compatibility and verification

- Pair each `pair{i}` with its matching `atom{i}`. The named route resolves
  both; these v1 pair checkpoints do not embed the atom submodel.
- The pair config requires `quadrupole_scale=1.5`. The unfused constructor and
  `set_pretrained_model` restore it. Loading only the state dict loses this
  forward-pass constant and produces incorrect electrostatics.
- Fused routes (`ap2_fused=True`) reject these separate atom/pair checkpoints.
- Compare the five-member ensemble, not one member, with the paper.
- Earlier converted pair checkpoints were incomplete, including uninitialized
  readout parameters. Replace them and discard predictions made with them.

`tests/test_ap2_tf_parity.py` checks all five members against recorded
TensorFlow 2.3 outputs without requiring TensorFlow at test time. Its 26 dimers
include charged and monatomic monomers and a following dimer to catch
batch-index corruption from edgeless atoms. The tests also reject missing,
non-finite, or lazy checkpoint tensors.

### Per-member prediction agreement

The original 24-dimer comparison measured the following maximum absolute
PyTorch–TensorFlow differences. These numbers describe the original subset,
not the two subsequently added monatomic/successor regression cases.
Multipoles use the 10-component charge/dipole/quadrupole vector; energies are
in kcal/mol.

| Member | Multipoles A max Δ | Multipoles B max Δ | SAPT component max Δ | Total energy max Δ |
|---|---|---|---|---|
| 0 | 5.96e-7 | 1.22e-6 | 6.48e-5 | 6.40e-5 |
| 1 | 5.96e-7 | 3.22e-6 | 9.16e-5 | 7.77e-5 |
| 2 | 5.96e-7 | 1.88e-6 | 8.33e-5 | 8.19e-5 |
| 3 | 7.75e-7 | 1.05e-6 | 1.21e-4 | 1.21e-4 |
| 4 | 5.96e-7 | 1.25e-6 | 5.34e-5 | 7.72e-5 |

Against the same 24 reference labels, total-energy MAE agrees to four decimals
for every member (kcal/mol):

| Member | PyTorch MAE | TensorFlow MAE |
|---|---|---|
| 0 | 0.2520 | 0.2520 |
| 1 | 0.3366 | 0.3366 |
| 2 | 0.2563 | 0.2563 |
| 3 | 0.2487 | 0.2487 |
| 4 | 0.1834 | 0.1834 |

### Full-validation ensemble agreement

The converted and TensorFlow ensembles gave the following MAEs on the paper's
150,000-dimer validation split (kcal/mol):

| Component | PyTorch | TensorFlow | Paper Fig. 2B |
|---|---|---|---|
| Elst | 0.16704 | 0.16704 | 0.168 |
| Exch | 0.14079 | 0.14079 | 0.141 |
| Ind | 0.09567 | 0.09567 | 0.096 |
| Disp | 0.02043 | 0.02043 | 0.021 |
| Total | 0.19992 | 0.19992 | 0.201 |

Mean absolute total prediction difference was 1.6e-5 kcal/mol; one dimer
exceeded 1e-2 on a component. Verification used CPU float32; reduction order
and cutoff-boundary behavior can differ across devices.

## Regenerating artifacts

Normal inference needs no TensorFlow. Conversion uses a legacy Python 3.8 /
TensorFlow 2.3.4 environment; its dependencies are recorded in
`scripts/ap2_tf/legacy-tf-env.pip-freeze.txt`.

```bash
scripts/ap2_tf/make_legacy_tf_env.sh

# Legacy environment: export SavedModel variables and provenance.
python scripts/ap2_tf/export_tf_savedmodel.py \
    <apnet>/apnet/atom_models/atom0 <apnet>/apnet/pair_models/pair0 \
    --out-dir tf_npz

# PyTorch environment: repeat for members 0–4.
python scripts/ap2_tf/convert_tf_to_pt.py --kind atom --npz tf_npz/atom0.npz \
    --out models/ap2_tf_paper/atom_models/atom0.pt --overwrite
python scripts/ap2_tf/convert_tf_to_pt.py --kind pair --npz tf_npz/pair0.npz \
    --out models/ap2_tf_paper/pair_models/pair0.pt --overwrite
```

The converter validates variable names, shapes, and complete consumption:
135 atom variables and 218 pair variables, including the embedded atom model.
To regenerate the regression fixture:

```bash
# PyTorch environment.
python scripts/ap2_tf/make_parity_dimers.py \
    --processed-dir <data_dir>/processed \
    --prefix dimer_ap2_fused_test_spec_2_ --samples 24 --ensure-monatomic \
    --out-npz tests/dataset_data/ap2_tf_parity/parity_dimers.npz \
    --out-manifest tests/dataset_data/ap2_tf_parity/parity_dimers.manifest.json

# Legacy environment.
python scripts/ap2_tf/tf_reference_predictions.py \
    --dimers-npz tests/dataset_data/ap2_tf_parity/parity_dimers.npz \
    --out-npz tests/dataset_data/ap2_tf_parity/tf_reference.npz \
    --out-manifest tests/dataset_data/ap2_tf_parity/tf_reference.manifest.json
```

`parity_common.py` preserves molecular coordinates across both environments.
`upload_paper_models_to_hf.py` publishes the checkpoints using the loader's
registry and verifies uploaded hashes; use `--dry-run` to preview.

See [training parity controls](specs/apnet2-tensorflow-parity.md) for retraining.
