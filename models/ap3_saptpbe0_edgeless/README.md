# AP3D3 SAPT/PBE0 stack, retrained on the edgeless-atom fix

The checkpoints in `models/ap3_saptpbe0/` were trained *through* the atom-model
edgeless bug described in the pull request that added this directory: whenever a
batch contained a monatomic monomer, `AtomMPNN` scattered dipole and quadrupole
messages under the unfiltered atom count while the edge indices had been
remapped into the filtered numbering, so every later atom in that batch received
another atom's multipoles. Those weights are therefore fitted to a forward pass
that no longer exists, and they are not interchangeable with the fixed code.

This directory holds the replacement stack, retrained from scratch on the fixed
implementation. It uses a new path rather than overwriting `ap3_saptpbe0/1/`, so
both generations remain available for comparison.

## What is here

| file | stage | model | trained on |
|---|---|---|---|
| `1/am_ap2_1.pt` | 1 | `AtomMPNN` (AP2 atom model) | PBE0 monomers, dataset spec 4 |
| `1/atp_hfvr_1.pt` | 2 | `AtomTypeParamNN`, `hirshfeld_volume_ratio__valence_width` | dataset spec 1 |

sha256:

```
fb0886e744cc77e36b25081a57807d1c9794f598c0254854a0dd6289479235a4  1/am_ap2_1.pt
4f8315ceb1bded66a72cb4b23efe1ef502759360692afe99d52dffc6299143c2  1/atp_hfvr_1.pt
```

Both are `checkpoint_version` 2 with the standard metadata block, so they load
through `model_io` like any other checkpoint:

```python
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamModel

am = AtomModel(
    ds_root=None, ignore_database_null=True,
    pre_trained_model_path="models/ap3_saptpbe0_edgeless/1/am_ap2_1.pt",
)
atp = AtomTypeParamModel(
    ds_root=None, ignore_database_null=True,
    atom_model_pre_trained_path="models/ap3_saptpbe0_edgeless/1/am_ap2_1.pt",
    pre_trained_model_path="models/ap3_saptpbe0_edgeless/1/atp_hfvr_1.pt",
)
```

## How they were produced

Both stages ran single-process (no DDP: at batch size 16 the AP2 recipe is what
matters, and `DistributedSampler` would have changed the effective batch), on
one V100, from `train_ap3d3_saptdft_local_1_edgeless_fix.sh`.

| | stage 1 | stage 2 |
|---|---|---|
| entry point | `train_models.py --train_am AtomModel` | `train_models.py --train_apnet AtomTypeParamModel` |
| dataset spec | 4 | 1 |
| epochs | 500 | 100 |
| lr | 5e-4, constant | 5e-5, constant |
| architecture | `n_message 3`, `n_rbf 8`, `n_neuron 128`, `n_embed 8` | `n_message 3`, `n_rbf 8`, `n_neuron 32`, `n_embed 8` |
| random seed | 1 | 1 |
| best epoch | 453 | 95 |
| best val loss sum | 0.03702 | 0.0079 |
| wall time | 06:45:51 | 00:08:07 |

Training curves are under the `ap3d3-edgeless-fix-retrain` W&B project, runs
`s1-am-ap2` and `s2-atp-hfvr`.

## Still to come

Stages 3-5 of the stack — the `AM_DimerParam` electrostatics model, the fused
`APNet3-fused-d3` model on Splinter, and its SAPT/PBE0 fine-tune — are training
now and will land in `1/` beside these two. Stage 2's output is the input to
both of the remaining `AtomTypeParam` consumers, which is why it ships first:
the branches that need an atom/hfvr pair trained on the fixed forward pass do
not have to wait for the full stack.
