# Rackers Thole Damping Model Design

## Summary

Add a dedicated atom-type model and training harness in
`src/apnet_pt/AtomPairwiseModels/mtp_mtp.py` that predicts three positive,
environment-dependent parameters per atom:

1. an electrostatics damping parameter,
2. a direct Thole damping parameter for permanent-induced interactions, and
3. a mutual Thole damping parameter for induced-induced interactions.

For each interacting atom pair, the direct and mutual Thole parameters use the
geometric combination rule

\[
K_{ij} = \sqrt{K_i K_j}.
\]

The model is trained end-to-end against SAPT electrostatics and induction
energies. It is exposed through `train_models.py` as
`RackersTholeDampingModel`.

## Goals

- Predict distinct electrostatic, direct-Thole, and mutual-Thole parameters for
  every atom.
- Guarantee that all three predicted parameters are positive and suitable for
  square-root combination.
- Apply direct and mutual Thole values to their physically distinct interaction
  tensors.
- Train the model jointly against SAPT electrostatics and induction components.
- Preserve all existing `AtomTypeParamNN`, `AM_DimerParam_Model`, and checkpoint
  behavior.
- Provide focused, CPU-compatible pytest coverage for model behavior, physics
  routing, training dispatch, and checkpoint round trips.

## Non-goals

- Do not introduce categorical atom-type labels or classification losses.
- Do not predict parameters directly on atom-pair edges.
- Do not change the output semantics of the existing `AtomTypeParamNN`.
- Do not migrate existing checkpoints to the new model.
- Do not alter the pairwise dataset schema or SAPT component ordering.
- Do not introduce component loss weights; electrostatics and induction retain
  equal MSE weighting.
- Do not refactor unrelated electrostatic or polarization implementations.

## Architecture

### Nested atom model

The new model wraps the existing pretrained `AtomTypeParamNN` used to provide:

- atomic charges,
- atomic dipoles,
- atomic quadrupoles,
- hidden-state history,
- Hirshfeld volume ratios, and
- valence widths.

The nested model is frozen by default. A caller can opt into end-to-end
fine-tuning through the existing `--unfreeze_atom_model` CLI flag. Freezing must
be enforced with `requires_grad_(False)` rather than relying only on optimizer
parameter selection.

### `RackersTholeDampingNN`

Add `RackersTholeDampingNN` in `mtp_mtp.py`. It owns exactly three independent
per-atom readout heads with stable column meanings:

| Column | Name | Meaning | Initial positive value |
| --- | --- | --- | ---: |
| 0 | `elst` | Electrostatic damping parameter | 1.8 |
| 1 | `thole_direct` | Permanent-induced Thole parameter | 0.34 |
| 2 | `thole_mutual` | Induced-induced Thole parameter | 0.39 |

The class must not accept a variable `n_params`; the output count and ordering
are part of its public and checkpoint contracts.

Each head predicts an unconstrained raw value. The public parameter is

\[
K = \operatorname{softplus}(K_{raw}) + \epsilon,
\]

where `epsilon` is exactly `1e-8` by default and is recorded in model
configuration. Initialization uses inverse softplus so that zero correction
output maps to the requested positive initial value, accounting for `epsilon`.
The three raw initialization standard deviations default to
`[0.01, 0.01, 0.01]`.

The forward return preserves the repository's tuple convention for nested atom
models. Multipoles and wrapped auxiliary outputs retain their current positions,
and the final item is always a tensor of shape `[n_atoms, 3]`. Existing
`AtomTypeParamNN` rank behavior remains unchanged.

### Geometric combination helper

Add a focused helper that accepts source and target per-atom values plus edge
indices and returns edge values:

\[
K_{ij} = \sqrt{K_i K_j}.
\]

The helper must:

- select source and target values before combining them,
- preserve floating-point dtype and device,
- be symmetric under exchanging source and target inputs,
- reject non-finite inputs with a clear `ValueError`, and
- rely on the model's positivity contract rather than silently taking absolute
  values.

This helper is used independently for direct and mutual Thole columns. The
existing electrostatic damping path continues to receive separate positive
per-atom values, because the geometric rule applies only to Thole damping.

### Rackers induction kernel

Add a Rackers-specific induction path rather than changing the defaults of the
existing induction kernels. Its inputs include separate positive per-atom
`thole_direct_A/B` and `thole_mutual_A/B` tensors.

For every intermolecular AB edge and every intramolecular AA and BB edge, the
kernel computes the appropriate geometric mean. It then applies:

- direct pair values to permanent-charge/permanent-dipole to induced-dipole
  interaction tensors, and
- mutual pair values to induced-induced interaction tensors used during the SCF
  iterations.

The same parameter semantics and combination logic must be used for AB, AA, and
BB edges. Existing Hirshfeld-scaled polarizabilities, overlap correction,
valence-width handling, SCF convergence settings, energy units, and energy
aggregation remain unchanged unless a minimal signature split is required to
route direct and mutual tensors correctly.

### `DimerProp` mode

Add a stable `DimerProp` mode named `rackers_thole`. Its forward path:

1. evaluates `RackersTholeDampingNN` for monomers A and B,
2. reads columns by named constants or explicit unpacking rather than magic
   negative indices,
3. computes damped electrostatics using column 0,
4. computes induction using columns 1 and 2 through the Rackers kernel, and
5. returns `torch.vstack((Elst, Indu)).T`, with shape `[n_edges, 2]`, plus the
   monomer output tuples.

Electrostatics must be evaluated without corrupting charge tensors needed by
induction. The implementation must clone charge inputs where necessary instead
of relying on mutation-sensitive evaluation order.

Training and prediction both use the full intermolecular edge domain:
`e_ABfull_source`, `e_ABfull_target`, and `dimer_ind_full`. If the target-bearing
collate path does not currently emit `dimer_ind_full`, it will derive that index
without changing stored dataset records. Intramolecular direct and mutual terms
use the existing full AA and BB edge sets.

### `RackersTholeDampingModel`

Add a dedicated high-level harness named `RackersTholeDampingModel`. It may
subclass or compose `AM_DimerParam_Model`, but it must expose the Rackers model
without duplicating complete training loops.

The harness fixes:

- low-level model type to `RackersTholeDampingNN`,
- dimer evaluation mode to `rackers_thole`,
- output count and ordering to the three named parameters, and
- training targets to SAPT columns `[0, 2]`, electrostatics and induction.

The joint loss is the existing mean-squared error over both selected component
columns with equal weighting. Prediction returns dimer-level electrostatics and
induction values after scatter aggregation.

## Checkpoint contract

Rackers checkpoints use the repository's current checkpoint format and embed
sufficient nested-model state and configuration for standalone reconstruction.
The saved configuration includes:

- model type `RackersTholeDampingNN`,
- parameter names in exact order,
- the three desired initial positive values,
- initialization standard deviations,
- positivity epsilon,
- readout architecture settings,
- nested atom-model configuration,
- `dimer_eval="rackers_thole"`, and
- electrostatic damping type.

Loading must validate that parameter names exist and exactly match
`("elst", "thole_direct", "thole_mutual")`. A missing or reordered parameter
contract raises `ValueError` rather than silently changing physical meanings.
Existing checkpoint reconstruction remains unchanged for non-Rackers models.

## Training route

Extend `train_pairwise_model` and CLI help in `train_models.py` with the exact
identifier `RackersTholeDampingModel`.

Canonical invocation:

```bash
python train_models.py \
  --train_apnet RackersTholeDampingModel \
  --am_model_path <multipole-model.pt> \
  --atom_type_param_model_path <hfvr-vw-model.pt> \
  --ap_model_path <output-model.pt> \
  --data_dir <pairwise-dataset>
```

The route first constructs or loads the existing HFVR/valence-width
`AtomTypeParamModel`, then wraps its low-level model in
`RackersTholeDampingModel`. It forwards dataset, architecture, learning-rate,
seed, checkpoint, and freeze settings through the existing pairwise training
flow.

Rackers mean defaults are `[1.8, 0.34, 0.39]`; Rackers raw standard-deviation
defaults are `[0.01, 0.01, 0.01]`. The CLI parser will use an unset sentinel for
`--param_start_mean` and `--param_start_std`, then resolve to these defaults only
for the Rackers route and preserve the existing defaults for all other routes.
Either existing option may override Rackers defaults only with exactly three
comma-separated values. Explicit scalar broadcasting remains available to
existing model routes but is rejected for the Rackers route so parameter intent
is unambiguous.

`--unfreeze_atom_model` enables fine-tuning of the nested model. Without that
flag, nested model parameters remain frozen. The generic `--n_params` argument
is ignored by other routes as before but must not configure the Rackers model;
the Rackers route always has three outputs.

## Error handling

Raise clear `ValueError`s for:

- initialization mean or standard-deviation lists that do not contain exactly
  three values,
- unsupported nested atom-model types,
- non-finite per-atom values presented to the geometric combination helper,
- Rackers checkpoints with absent, reordered, or unknown parameter metadata,
- incompatible checkpoint model types, and
- unknown `DimerProp` modes through the existing dispatcher behavior.

The model's softplus transform guarantees positive outputs. Downstream Rackers
code must not use `abs` or clamping to conceal violations of this contract.

## Testing strategy

Create `tests/test_rackers_thole_damping.py` with focused tests that do not
require network or database access.

### Parameter-head test

Using a lightweight deterministic nested atom-model fixture, verify:

- output shape is `[n_atoms, 3]`,
- outputs are finite and strictly positive,
- zeroed correction heads initialize near `[1.8, 0.34, 0.39]`,
- wrapped multipole and auxiliary outputs are preserved, and
- backward propagation produces finite gradients for all three readout heads.

### Combination-rule test

Use different source and target values to verify exact geometric means for
multiple edges. Verify symmetry when source/target sets and edge indices are
exchanged. Verify non-finite input rejection.

### Physics-routing test

Use controlled parameter values or monkeypatched damping helpers to verify that:

- direct geometric means are passed only to direct Thole damping,
- mutual geometric means are passed only to mutual Thole damping,
- AB, AA, and BB edge sets use the same rule, and
- electrostatic damping receives the separate electrostatic parameter column.

### Joint-forward and gradient test

Build a small CPU-only synthetic dimer batch and verify:

- the Rackers dimer forward returns finite `[n_edges, 2]` values,
- dimer scatter aggregation produces finite `[batch_size, 2]` values,
- backward propagation reaches all three heads, and
- outputs remain positive after one optimizer step.

### Training-dispatch test

Monkeypatch heavyweight dataset/model construction and verify that
`train_pairwise_model(apnet_model_type="RackersTholeDampingModel", ...)`:

- selects the new harness,
- supplies `[1.8, 0.34, 0.39]` by default,
- forwards the multipole and HFVR/valence-width checkpoint paths,
- freezes the nested atom model by default,
- honors the unfreeze option, and
- invokes the harness training method with supported arguments.

### Checkpoint round-trip test

Save and reload a small Rackers model, then verify:

- predictions match within floating-point tolerance,
- parameter metadata and ordering are preserved,
- nested state is restored without requiring a separately supplied compatible
  object, and
- an intentionally reordered parameter-name configuration is rejected.

## Verification commands

Run at minimum:

```bash
python -m pytest tests/test_rackers_thole_damping.py -v
python -m pytest tests/test_atomtype_props.py::test_elst_multipoles_MTP_torch_damping_AM_DimerParam -v
python -m pytest tests/test_polarization.py -k "thole" -v
python train_models.py --help
```

If the focused tests expose shared checkpoint or dataset behavior, also run the
relevant focused tests from `tests/test_model_io.py` and
`tests/test_pt_dataset.py`.

## Acceptance criteria

The feature is accepted when:

1. `RackersTholeDampingNN` returns three positive, correctly initialized
   per-atom parameters with stable documented meanings.
2. Direct and mutual edge values are computed as geometric means and routed to
   their corresponding Thole interaction tensors.
3. Joint electrostatics/induction prediction and backpropagation are finite on a
   CPU-only synthetic dimer.
4. `--train_apnet RackersTholeDampingModel` selects and trains the new harness
   while freezing its nested atom model by default.
5. Rackers checkpoints round-trip with parameter metadata and nested state
   intact.
6. The new focused tests and existing selected damping regressions pass.
7. Existing model identifiers and checkpoint behavior remain unchanged.
