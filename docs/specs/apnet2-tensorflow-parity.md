# APNet2 TensorFlow/PyTorch parity controls

The original TensorFlow APNet2 and the QCMLForge PyTorch implementation share
the same high-level architecture, but they do not share every numerical training
default. The differences below are accuracy-sensitive and are exposed as explicit
controls so experiments can change one factor at a time.

## Confirmed differences

| Setting | TensorFlow APNet2 | Historical QCMLForge APNet2 |
|---|---|---|
| Pair dense kernels | Keras Glorot uniform | PyTorch `Linear` default |
| Pair element embedding | Uniform in `[-0.05, 0.05]` | PyTorch `Embedding` default |
| Adam epsilon | `1e-7` | `1e-8` |
| Saved pair checkpoint | Lowest validation total-energy MAE | Lowest validation component MSE |
| Quadrupoles in analytic electrostatics | Multiply by `3/2` | No multiplier |

Both implementations optimize an unweighted MSE over the four SAPT components,
use batch size 16, learning rate `5e-4`, three message-passing iterations, eight
radial functions, 128 base neurons, eight-dimensional embeddings, a 5 Å
intramonomer cutoff, and an 8 Å pair cutoff. The published interaction model was
trained for 50 epochs.

## PyTorch controls

`train_models.py` exposes the parity choices without changing historical defaults:

```bash
python train_models.py \
  --train_apnet APNet2-fused \
  --quadrupole-scale 1.5 \
  --parameter-initialization tensorflow \
  --adam-eps 1e-7 \
  --checkpoint-metric total_mae
```

The corresponding historical baseline is:

```bash
python train_models.py \
  --train_apnet APNet2-fused \
  --quadrupole-scale 1.0 \
  --parameter-initialization pytorch \
  --adam-eps 1e-8 \
  --checkpoint-metric component_mse
```

The quadrupole and initialization policy are saved in new checkpoints. Legacy
checkpoints that do not contain those fields retain the historical PyTorch
behavior (`quadrupole_scale=1.0`, `parameter_initialization="pytorch"`). The
TensorFlow initialization policy applies only to the APNet2 pair module; it does
not reinitialize the frozen atomic model.

Use `--ds_max_size` to cap pairwise datasets as well as atomic datasets. Use
`--wandb-run-config path.json` to merge immutable dataset/campaign provenance
into the W&B run configuration. Computed model/training facts take precedence
over colliding user-supplied keys.

## Controlled comparison protocol

1. Keep the atomic checkpoint, processed graph shards, train/validation identities,
   seed, batch size, learning rate, epoch count, and metric code fixed.
2. Screen each setting above one at a time against the historical baseline.
3. Compare the fully TensorFlow-aligned arm against the strongest one-factor arm.
4. Repeat leading arms with multiple seeds before attributing an accuracy change.
5. Evaluate selected checkpoints on the same locked Splinter validation subset and
   the same PDB13K molecule cache. Do not call a 13,326-row local PDB cache the
   paper-exact 13,216-row benchmark without an identity manifest.

Exact retraining is still not guaranteed across frameworks because the original
TensorFlow code seeds NumPy shuffling but does not seed TensorFlow parameter
initialization. TensorFlow/Keras and PyTorch Adam kernels can also differ beyond
identical public hyperparameters.
