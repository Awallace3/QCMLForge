# APNet2 training parity controls

For inference with the published ensemble, use the
[paper weights route](../apnet2-tensorflow-weights.md). The controls below
support retraining comparisons without changing historical defaults.

| Setting | TensorFlow APNet2 | Historical QCMLForge |
|---|---|---|
| Pair dense kernels | Glorot uniform | PyTorch `Linear` default |
| Pair embedding | Uniform `[-0.05, 0.05]` | PyTorch `Embedding` default |
| Adam epsilon | `1e-7` | `1e-8` |
| Checkpoint selection | Validation total-energy MAE | Validation component MSE |
| Electrostatic quadrupole scale | `1.5` | `1.0` |

```bash
python train_models.py \
  --train_apnet APNet2-fused \
  --quadrupole-scale 1.5 \
  --parameter-initialization tensorflow \
  --adam-eps 1e-7 \
  --checkpoint-metric total_mae \
  --deterministic
```

The historical settings are `--quadrupole-scale 1.0`,
`--parameter-initialization pytorch`, `--adam-eps 1e-8`, and
`--checkpoint-metric component_mse`. Both implementations optimize unweighted
four-component MSE; the published pair model used batch size 16, learning rate
`5e-4`, and 50 epochs.

Initialization affects only the pair module, not the frozen atomic model.
New checkpoints record initialization and electrostatics policy; legacy
checkpoints retain historical defaults when fields are absent. Quadrupole
scaling changes inference, not just training, and must match the checkpoint.

## Optional higher-order electrostatics

`--elst-include-uQ-QQ` restores the dipole-quadrupole and
quadrupole-quadrupole terms from upstream before commit `593d655`.
The published kernel omits both, so the flag defaults to false.
Quadrupole scaling enters these terms linearly and quadratically, respectively.
Changing either electrostatics control without retraining perturbs the
analytic contribution against which the short-range network was fitted.
`tests/test_ap2_elst_uq_qq.py` checks the restored kernel against the upstream
formula.

## Reproducible comparisons

- Keep dataset identities, atom checkpoint, seed, batch size, learning rate,
  epoch count, and evaluation code fixed; vary one setting at a time.
- `--deterministic` requests deterministic PyTorch kernels and sets the cuBLAS
  workspace. Unsupported operations warn rather than abort, so check logs.
  Seeding alone does not make CUDA scatter reductions deterministic.
- Single-process shuffling uses a dedicated generator seeded by `random_seed`,
  keeping initialization RNG consumption from changing batch order.
- Compare fixed epochs when studying checkpoint selection, and use multiple
  seeds: deterministic execution does not remove seed-to-seed variability.
- `--ds_max_size` caps datasets; `--wandb-run-config path.json` adds campaign
  provenance. Computed model/training facts override colliding supplied keys.

Matching these settings does not guarantee identical cross-framework training:
TensorFlow's original initialization was not seeded, and optimizer kernels
can differ.
