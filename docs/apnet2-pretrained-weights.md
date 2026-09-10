# APNet2 pretrained weight sets

`apnet2_model_predict`, `atom_model_predict`, and `set_pretrained_model` all take
a `weights=` keyword naming one of three published ensembles. All three live in
the [`awallace3/qcmlforge`](https://huggingface.co/awallace3/qcmlforge) Hugging
Face repository and are downloaded on demand.

| `weights=` | Hugging Face paths | Members | Total MAE | What it is |
|---|---|---|---|---|
| `"qcmlforge"` (default) | `qcmlforge/atom_models/am_{0..4}.pt`, `qcmlforge/pair_models/ap2_{0..4}.pt` | 5 | **0.2043** | Trained by this project on the paper's hyperparameters, on top of the corrected `AtomMPNN` |
| `"qcmlforge_v1"` | `am_ensemble/am_{0..4}.pt`, `ap2_ensemble/ap2_{0..4}.pt` | 5 | 0.4351 | The previous default. Its atom models predate the `AtomMPNN` scatter fix |
| `"ap2_tf_paper"` | `ap2_tf_paper/{atom,pair}_models/*.pt` | 5 | 0.2000 | The authors' released TensorFlow models, converted. See [apnet2-tensorflow-weights.md](apnet2-tensorflow-weights.md) |

Total MAE is on the 150 000-dimer Splinter test split (`test150k`) used for
Fig. 2B of Glick et al., *Chem. Sci.* **2024**, 15, 13313, which reports 0.201
kcal/mol for the five-member ensemble.

Nothing about the calling convention changed: `weights` still defaults to
`"qcmlforge"`, so existing code picks up the new weights automatically.
`"qcmlforge_v1"` reproduces results obtained before this change.

## Why the default moved

`AtomMPNN`'s message-passing readout used a scatter operation that dropped
contributions, which biased the predicted atomic multipoles and therefore the
electrostatics channel that consumes them. The fix is a correctness fix, not a
tuning change, and every model that reads multipoles inherits it.

Re-evaluating the *old* default ensemble with the fixed forward pass improves it
(0.4695 → 0.4351) but does not repair it: those atom models were trained against
the buggy pass, so their weights encode the bug. Only retraining does, which is
what the `"qcmlforge"` set is.

| | Total | Elst | Exch | Ind | Disp | max abs Total err | % within 1 kcal/mol |
|---|---|---|---|---|---|---|---|
| `qcmlforge_v1`, evaluated before the fix | 0.4695 | 0.4500 | 0.1752 | 0.1173 | 0.0246 | 23.78 | 87.53 |
| `qcmlforge_v1`, evaluated after the fix | 0.4351 | 0.4168 | 0.1752 | 0.1173 | 0.0246 | 20.74 | 88.56 |
| **`qcmlforge` (new default)** | **0.2043** | 0.1694 | 0.1448 | 0.0985 | 0.0225 | 19.66 | 97.21 |
| `ap2_tf_paper` (authors' weights) | 0.2000 | 0.1670 | 0.1408 | 0.0957 | 0.0204 | 14.30 | 97.32 |
| paper Fig. 2B | 0.201 | 0.168 | 0.141 | 0.096 | 0.021 | <15 | >97% |

Ensemble MAE, kcal/mol, 150 000 dimers. Exch/Ind/Disp are identical between the
two `qcmlforge_v1` rows because those channels do not consume multipoles; the
whole difference is in Elst.

The new default reaches the paper's ensemble number to 0.003 kcal/mol and passes
the paper's "97% within 1 kcal/mol" gate. It does not pass the "no error above
15 kcal/mol" gate: one dimer of the 150 000 is off by 19.66.

## Provenance of the default set

Trained with the paper's §3.2 recipe: `n_message=3`, `n_neuron=128`,
`n_embed=8`, `n_rbf=8`, `r_cut=5.0`, `r_cut_im=8.0`, batch size 16, Adam at a
constant 5e-4, 50 epochs, `quadrupole_scale=1.5`, and the weights of the epoch
with the lowest validation **MSE** (not MAE — this is the paper's rule and it
matters; see below). Training data is the AP-Net2 SAPT0/aug-cc-pV(D+d)Z Splinter
set: 53 173 train / 47 855 in-set / 5 318 validation dimers.

Each member is an atom model trained first, then a pair model trained against
it. W&B project `ap2-tf-paper-repro`, entity
`awallace43-georgia-institute-of-technology`:

| `model_id` | W&B run | Member Total MAE | Saved epoch |
|---|---|---|---|
| 0 | `nbedmu31` | 0.2926 | 46 |
| 1 | `qjqp3o7w` | 0.2885 | 44 |
| 2 | `keen7mtv` | 0.2870 | 50 |
| 3 | `huwhzcza` | 0.2918 | 50 |
| 4 | `vt2ali2m` | 0.3126 | 47 |

Each `qcmlforge/pair_models/ap2_{i}.pt` is a v2 checkpoint that **embeds its own
atom model** under `submodels.atom_model`; that is the 10.2 MB vs 4.0 MB file
size difference against `qcmlforge_v1`. The pair network itself is the same 83
tensors / 998 012 parameters. The embedded submodel is tensor-by-tensor
bit-identical (`torch.equal`) and config-identical to the separately published
`qcmlforge/atom_models/am_{i}.pt`, verified for all five members, so the two
routes below give identical results:

```python
# Embedded atom model, one file:
APNet2Model(pre_trained_model_path="ap2_0.pt")
# Explicit atom model, two files — same weights, no warning:
APNet2Model(pre_trained_model_path="ap2_0.pt", atom_model_pre_trained_path="am_0.pt")
```

Passing a *different* atom model alongside an embedded one still warns and still
uses the embedded one; `model_io.embedded_submodel_matches_external` is what
distinguishes the two cases.

## Reading the numbers: one member is not the ensemble

A single member scores 0.287–0.313, not 0.201. Comparing one training run
against the paper's headline number overstates the gap by ~1.4x. The ensemble
gain follows one parameter, the mean pairwise correlation of member errors:

`MAE(n) = MAE(1) * sqrt(rho + (1 - rho) / n)`

| | MAE(1) | rho | measured n=5 | law n=5 | realized gain |
|---|---|---|---|---|---|
| `qcmlforge` | 0.2945 | 0.3625 | 0.2043 | 0.2062 | 30.6% |
| `ap2_tf_paper` | 0.2873 | 0.3528 | 0.2000 | 0.1995 | 30.4% |

Measured `n`-member sweep for the default set (mean over all subsets of size
`n`): 0.2945 (n=1), 0.2424 (2), 0.2221 (3), 0.2112 (4), 0.2043 (5).

Decorrelation is therefore *not* where the default set falls short of the
authors': our ensembling gain slightly exceeds theirs. The entire 0.0043
ensemble gap is the 0.0072 single-member gap, and 0.0042 of that — 58% — is the
paper's MSE-based checkpoint selection rule, which picked a worse-MAE epoch than
the best available one for three of five members. That rule is what the paper
did, so it stays.

The same shortfall appears when the reference TensorFlow implementation is
trained on this pipeline (0.2949), so it is not an artifact of the PyTorch port.

## Publishing a new set

`scripts/ap2_tf/upload_paper_models_to_hf.py` uploads whatever the registry in
`src/apnet_pt/hf_pretrained.py` says a weight set contains, then re-downloads
each file and compares sha256, so a partial or corrupted commit fails loudly:

```bash
python scripts/ap2_tf/upload_paper_models_to_hf.py \
    --weights qcmlforge --models-dir /path/to/staged --dry-run
```

`--models-dir` points at a directory laid out like the remote paths with the
weight-set name stripped, i.e. `atom_models/am_{i}.pt` and
`pair_models/ap2_{i}.pt`. Drop `--dry-run` to commit. Add the weight set to
`_PRETRAINED_MODEL_GROUPS` in `tests/conftest.py` so tests that need it skip
cleanly when downloads are disabled.

## Not covered by this weight set

The AP3 and DAPNet2 code paths hard-code `am_ensemble/am_0.pt` and
`dapnet2/backbone/*` rather than going through the registry. They still resolve
the `qcmlforge_v1` atom models. Repointing them needs an AP3 evaluation against
the retrained atom model, which does not exist yet.
