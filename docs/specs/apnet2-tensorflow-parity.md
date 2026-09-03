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

The quadrupole multiplier is the one entry that is not merely a training
default: it changes the forward pass, so it must be set correctly to *evaluate*
a TensorFlow-derived checkpoint, not only to retrain like one. See
[Running APNet2 with the original TensorFlow weights](../apnet2-tensorflow-weights.md).

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

## Pre-rewrite electrostatics terms

`apnet` commit 593d655 (2021-07-18, "full rewrite") made exactly one numerical
change to the analytic multipole electrostatics: it dropped the
dipole-quadrupole and quadrupole-quadrupole terms from the interaction sum. The
pre-rewrite routine summed `qq + qu + qQ + uu + uQ + QQ`, the published one sums
`qq + qu + qQ + uu`. Every commit after it leaves the kernel's arithmetic alone,
and the shipped SavedModels postdate it, so the published weights were trained
against the four-term kernel.

The `3/2` quadrupole prefactor is *not* a change introduced by that commit. The
pre-rewrite `apnet/multipoles.py` already multiplies `qpole_redundant(...)` by
`3.0/2.0` for both reference and predicted quadrupoles before evaluating the
interaction (`origin/master` = `593d655^` = `e09955b`, lines 129/133/207/211).
It is a convention conversion for the stored `cartesian_multipoles` layout that
both eras apply; 593d655 only moved it inside the traced graph. `1.5` is
therefore the right scale for every TensorFlow code era, and `1.0` corresponds
to none of them.

`elst_include_uQ_QQ` restores the two dropped terms, defaulting to `False` so
the published kernel stays the default:

```bash
python train_models.py --train_apnet APNet2-fused --elst-include-uQ-QQ
```

The restored terms use the pre-rewrite `T3`/`T4` Cartesian interaction tensors
verbatim, including the fact that the original `T3` is not fully index
symmetric — contracted against a traceless symmetric quadrupole the trace term
vanishes and the doubled term equals the two distinct symmetric ones, so the
contraction is unchanged. `tests/test_ap2_elst_uq_qq.py` pins the PyTorch
implementation to a transcription of that routine and asserts the flag changes
nothing but the electrostatics.

The flag interacts with `--quadrupole-scale`, which is applied to both
quadrupole tensors before any term is formed: it therefore enters `uQ` linearly
and `QQ` quadratically. `(1.5, off)` is the published kernel and `(1.5, on)` is
the pre-rewrite functional form. Both are forward-pass constants, so changing
either without retraining perturbs a model whose learned short-range readout
adapted to the kernel it was trained with. Only the pairs beyond the 8 A cutoff,
which receive no readout at all, measure the analytic term unaided.

## Where the converted weights stand against the paper

The five converted TensorFlow members, averaged the way
`apnet/bms_functions.py::predict_sapt` averages them, evaluated on the paper's
own 150 000-dimer Splinter validation split:

| component | converted ensemble MAE | paper Fig. 2B | delta |
|---|---|---|---|
| Elst | 0.3024 | 0.168 | **+0.134** |
| Exch | 0.1408 | 0.141 | -0.0002 |
| Ind | 0.0957 | 0.096 | -0.0003 |
| Disp | 0.0204 | 0.021 | -0.0006 |
| Total | 0.3293 | 0.201 | +0.128 |

Exchange, induction and dispersion are parity-verified: they land 2-6e-4
kcal/mol from the published ensemble. Because none of the shared machinery is
component-specific, that also verifies the conversion, the architecture, the
featurisation, the ensemble rule and the subset identity. Electrostatics is the
only open discrepancy.

Two rules follow for anyone comparing to the paper:

- **Compare ensembles, never members.** Averaging the five members is worth
  0.070 kcal/mol on the total and 0.048 on Elst. Every published AP-Net2 number
  is a five-model average, so a single-member MAE is not comparable to it.
- **Use `quadrupole_scale=1.5`.** It is a forward-pass constant rather than a
  state-dict entry, so a loader that only calls `load_state_dict` reports
  success and silently evaluates the wrong electrostatics. Setting it to 1.0
  costs 0.012 kcal/mol of Elst and corresponds to no TensorFlow code era.

The Elst gap is not generalisation -- the training split gives 0.294 against
the validation split's 0.302 -- and it is not the electrostatics kernel's
history: re-pairing the atom models (all 20 off-diagonal combinations),
dropping the `3/2`, and restoring the `uQ`/`QQ` terms are worth +0.011, +0.012
and +0.0008 respectively, all in the wrong direction and all an order of
magnitude too small.

It is also not the multipole source. `PairModel.pretrained(i)` restores
`pair{i}` as one SavedModel with its atom network embedded by
`KerasPairModel(atom_model.model)`, while the conversion took its multipole
weights from the standalone `atom_models/atom{i}`, so the two could in
principle be different networks. They are not:
`tests/test_ap2_tf_parity.py::test_pair_model_reproduces_tensorflow` compares
the converted pair network fed by the standalone atom model against components
TensorFlow itself produced through the embedded submodel, and agrees to
1.2e-4 kcal/mol on electrostatics across 24 dimers and all five members.
Electrostatics consumes the multipoles analytically, so that agreement is only
possible if the two multipole sources are the same network.

What that leaves is a discrepancy between the *shipped TensorFlow SavedModels*
and the paper's reported electrostatics MAE, not between TensorFlow and
QCMLForge. The conversion reproduces the SavedModels' own forward pass to
~1e-4 kcal/mol; evaluating those SavedModels in TensorFlow on the same
150 000-dimer subset would confirm the published checkpoints themselves give
0.30 rather than 0.168, and is the experiment that would close this out. Until
then, describe these checkpoints as reproducing the published SavedModels'
predictions -- which is verified -- and not as reproducing the paper's
reported electrostatics error.

## Controlled comparison protocol

1. Keep the atomic checkpoint, processed graph shards, train/validation identities,
   seed, batch size, learning rate, epoch count, and metric code fixed.
2. Screen each setting above one at a time against the historical baseline.
3. Compare the fully TensorFlow-aligned arm against the strongest one-factor arm.
4. Repeat leading arms with multiple seeds before attributing an accuracy change.
5. Evaluate selected checkpoints on the same locked Splinter validation subset and
   the same PDB13K molecule cache. Do not call a 13,326-row local PDB cache the
   paper-exact 13,216-row benchmark without an identity manifest.

## Seed choice sets the resolution floor

GPU training of this model is not reproducible by seeding alone. APNet2 message
passing accumulates edge contributions with `Tensor.scatter_add_`, which on CUDA
sums in nondeterministic thread order. Two runs of an identical configuration
therefore diverge, starting near float32 epsilon and amplifying through
training.

The `ap2-tf-parity-screen-10k-v1` campaign measured this by accident. Its
`pt-baseline` and `tf-checkpoint` arms differ only in `--checkpoint-metric`,
which changes which epoch is saved and not the training math, so their
trajectories are an unintended replicate pair. Observed validation loss:

| Epoch | `pt-baseline` | `tf-checkpoint` | Relative gap |
|---|---|---|---|
| 0 | 82695.568 | 82695.564 | 5e-8 |
| 1 | 13152.473 | 12827.399 | 2.5% |
| 10 | 2197.929 | 1800.070 | 18% |

`--deterministic` removes that divergence. It requests deterministic
`scatter_add_` and `gather` backward kernels and pins the cuBLAS workspace. Two
three-epoch runs of one configuration, submitted as separate array tasks that
landed on different V100 nodes, produced zero differences across every logged
metric and the same checkpoint SHA-256. Neither run emitted a `warn_only`
fallback warning, so every op had a deterministic kernel and `torch.compile` did
not reintroduce a nondeterministic reduction. Cost: about 4% per epoch (75.2 s
against 72.3 s), with epoch 0 unchanged at ~154 s because compilation dominates
it. Verify this per configuration rather than assuming it — the compiled graph,
not just the eager ops, has to cooperate.

What determinism does not buy is resolution. Rerunning the same six arms with
`--deterministic` and three seeds each gives a pooled within-arm standard
deviation of **0.126 kcal/mol** in validation total MAE at 10 epochs on 10K
samples — essentially the 0.130 kcal/mol the nondeterministic replicate pair
showed, and wider than the v1 screen's entire six-arm range of 0.213. Every
one-factor effect in that rerun is at most 0.4x the seed spread when arms are
compared at a fixed epoch. None of the four differences above is resolvable at
this scale.

Consequences for the protocol:

- Pass `--deterministic` for any comparison meant to isolate one factor. It
  makes a run reproducible and reduces the noise to a single term, seed choice,
  that replicates can actually estimate.
- Compare arms at a fixed epoch, not at each arm's selected checkpoint, whenever
  `--checkpoint-metric` differs between them. Two arms that select on the
  ranking metric otherwise get a selection advantage over arms that do not.
- Read every delta as a multiple of the pooled within-arm spread. Separating a
  difference `d` at two standard errors needs roughly `n >= 8 sigma^2 / d^2`
  seeds per arm: about 50 for `d = 0.05` at `sigma = 0.126`, about 13 for
  `d = 0.10`. If an effect needs that many seeds, buy scale or epochs instead.

## Shuffling determinism

Single-process pair training gives its shuffling sampler a generator seeded from
`random_seed`. Without a dedicated generator, `RandomSampler` reseeds from the
global torch RNG at every epoch, so any change that consumes a different number
of global draws before the loader is built — most importantly
`--parameter-initialization tensorflow` — also changes per-epoch batch order.
That coupling makes an initialization ablation a two-factor comparison. The
distributed path is unaffected because `DistributedSampler` derives its
permutation from its own epoch seed.

Exact retraining is still not guaranteed across frameworks because the original
TensorFlow code seeds NumPy shuffling but does not seed TensorFlow parameter
initialization. TensorFlow/Keras and PyTorch Adam kernels can also differ beyond
identical public hyperparameters.
