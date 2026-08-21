# CLIFF Classical Exchange and CLIFF-2 Inference Design

## Summary

Add the CLIFF classical exchange-repulsion term to
`src/apnet_pt/AtomPairwiseModels/mtp_mtp.py` and a new inference-only
assembly model in `src/apnet_pt/AtomPairwiseModels/cliff_2.py`.

Exchange follows CLIFF Eq. (8) with the Van Vleet effective-width overlap of
Eq. (11):

\[
E^{\mathrm{exch}} = \sum_{i \in A}\sum_{j \in B} K_i^{\mathrm{exch}}
K_j^{\mathrm{exch}}\, S_{ij},
\qquad
S_{ij} = \left[\tfrac{1}{3}(B_{ij} r_{ij})^2 + B_{ij} r_{ij} + 1\right]
e^{-B_{ij} r_{ij}},
\qquad
B_{ij} = \frac{1}{\sqrt{\sigma_i \sigma_j}} .
\]

Unlike Thole damping, exchange uses the **multiplicative** combination rule
\(K_{ij} = K_i K_j\), matching CLIFF's \(K^{\mathrm{exch}}\),
\(K^{\mathrm{indu}}\), and \(K^{\mathrm{disp}}\) parameters.

Per-atom \(K_i^{\mathrm{exch}}\) reuses the existing two-part parameterization
in `AtomTypeParamNN`: a per-element base value held in a `NoisyConstantEmbedding`
indexed by \(Z\) (`self.guess_layer[p]`, `mtp_mtp.py:964`) plus a per-message-step
MPNN correction (`self.param_readout_layers[p]`, `mtp_mtp.py:978`). This is a
strict generalization of CLIFF's 17 element-plus-coordination-number atom types:
the embedding supplies the element-level parameter, and the MPNN correction
supplies the local-environment dependence that CLIFF encodes discretely by
coordination number. No categorical atom-type labels or classification loss are
introduced.

Three training routes are added:

1. `CliffExchangeModel` — exchange alone, fit to SAPT `Exch`.
2. `CliffClassicalModel` — elst + exch + induction jointly (pure
   induced-point-dipole induction).
3. `CliffClassicalOverlapModel` — the same three terms with the short-range
   induction overlap correction enabled.

`cliff_2.py` then adds `CLIFF2Model`, an inference-only model that combines a
trained classical parameter set with `qcml_dftd3` dispersion to emit a full
four-component SAPT0 prediction plus total. It serves both as an MPNN-
parameterized advanced force field and as the classical baseline that AP3-D3
must outperform.

## Overlap-term verification (blocking prerequisite)

The repository contains two disagreeing implementations of \(S_{ij}\), and this
design depends on resolving them.

| Location | \(B_{ij}\) | Extra scaling |
| --- | --- | --- |
| `mtp_mtp.py:2472` (Rackers induction overlap) | `torch.sqrt(1.0 / (sigma_A * sigma_B))` | none (`h2kcalmol` applied to the energy) |
| `mtp_mtp.py:2594`, `mtp_mtp.py:3053` | `torch.sqrt(1.0 / (sigma_A * sigma_B))` | none |
| `apnet3.py:295` (`valence_width_exch`) | `1.0 / (sigma_A * sigma_B)` | `hartree2kcal` folded into `S_ij` |

CLIFF Eq. (10) is typeset as \(B_{ij} = 1/(\sigma_i \sigma_j)\), which
`apnet3.py` reproduces literally. That reading is **not** dimensionally or
numerically viable. Substituting the paper's own Table I and Fig. 5 values for
a water-dimer hydrogen bond (\(K^{\mathrm{exch}}_{O2}=5.8538\),
\(K^{\mathrm{exch}}_{HO}=0.5996\), \(\sigma_O=0.39\), \(\sigma_H=0.36\),
\(r=1.95\) Å):

| Form | \(B_{ij}\) (bohr⁻¹) | \(B_{ij} r_{ij}\) | \(S_{ij}\) | pair \(E^{\mathrm{exch}}\) (kcal/mol) |
| --- | ---: | ---: | ---: | ---: |
| \(1/(\sigma_i\sigma_j)\) | 7.1225 | 26.25 | 1.03e-09 | 2.26e-06 |
| \(1/\sqrt{\sigma_i\sigma_j}\) | 2.6688 | 9.83 | 2.31e-03 | **5.08** |

SAPT0 exchange for the water dimer is ≈ +4.4 kcal/mol and is dominated by this
contact. Only \(B_{ij} = 1/\sqrt{\sigma_i \sigma_j}\) reproduces it; the literal
form underpredicts by six orders of magnitude. The `mtp_mtp.py` induction-overlap
implementation is therefore correct, and `apnet3.py:295` is missing a square
root.

Consequences for this design:

- The new shared helper uses \(B_{ij} = 1/\sqrt{\sigma_i \sigma_j}\).
- `apnet3.py:295` numerics are **not** corrected here. In `apnet3.py`, `S_ij`
  is multiplied by a learned `readout_layer_exch_quotient`, so the missing
  square root is absorbed into fitted weights; changing it would silently
  invalidate every trained AP3 checkpoint. Instead, replace the stale
  `# TODO: Implement valence width exchange` comment with a note stating that
  the function is a learned shape factor with a deliberately non-physical
  \(B_{ij}\), and pin its current output with a regression test so a future
  reader does not "fix" it in place.
- The physically correct helper is never routed into `apnet3.py`.

## Goals

- Provide one shared, tested, dimensionless \(S_{ij}\) helper used by classical
  exchange and by every existing induction-overlap call site.
- Predict a positive per-atom \(K_i^{\mathrm{exch}}\) from the existing nested
  `AtomTypeParamNN`, using the multiplicative pair rule.
- Train exchange standalone against SAPT `Exch`.
- Train elst + exch + induction jointly, with optional CLIFF Eq. (23)
  component/total loss weighting.
- Assemble trained classical terms plus `qcml_dftd3` dispersion into a single
  inference model producing four SAPT0 components and a total.
- Reduce redundant work: compute intermolecular distances once per forward pass
  and share them across elst, exch, and induction.
- Preserve all existing `AtomTypeParamNN`, `RackersTholeDampingNN`,
  `AM_DimerParam_Model`, `apnet3`, and checkpoint behavior.
- Provide CPU-only pytest coverage including a golden analytic \(S_{ij}\) value.

## Non-goals

- Do not implement CLIFF's discrete atom-type table (C4/C3/C2/N3/.../Br) or
  coordination-number typing; per-atom regression replaces it.
- Do not implement CLIFF's own Tang–Toennies dispersion (Eq. 18); dispersion is
  `qcml_dftd3`.
- Do not fit \(K^{\mathrm{disp}}\) or the Tang–Toennies \(x_{ij}\) argument.
- Do not correct or re-fit `apnet3.valence_width_exch` numerics.
- Do not add a training loop to `cliff_2.py`; it is inference-only.
- Do not change the pairwise dataset schema or SAPT component ordering
  (`y = [Elst, Exch, Ind, Disp]`, from `ap2_fused_ds.py:1201`).
- Do not migrate existing Rackers checkpoints to the new parameter contracts.
- Do not change `RACKERS_PARAMETER_NAMES` or the four-column Rackers ordering.

## Architecture

### Shared overlap helper

Add to `mtp_mtp.py`:

```python
def atomic_overlap_S_ij(
    valence_widths_A: torch.Tensor,
    valence_widths_B: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    dR_AB: torch.Tensor,
    width_floor: float = OVERLAP_WIDTH_FLOOR,
) -> torch.Tensor:
```

It returns the dimensionless per-edge \(S_{ij}\) and must:

- gather widths with a single `index_select` per monomer, never forming an
  \([n_A, n_B]\) outer product,
- floor widths with `clamp_min(width_floor)` rather than `torch.where`,
- compute `B_ij = torch.rsqrt(sigma_i * sigma_j)`, avoiding a separate
  reciprocal and square root,
- evaluate `x = B_ij * dR_AB` exactly once and use the Horner form
  `(x * (x / 3.0 + 1.0) + 1.0) * torch.exp(-x)`, saving one multiply versus the
  literal `x**2 / 3 + x + 1`,
- apply **no** unit conversion, so callers own `constants.h2kcalmol`,
- contain no data-dependent control flow, keeping it `torch.compile`-safe,
- preserve input dtype and device.

`OVERLAP_WIDTH_FLOOR` defaults to `0.1`, matching the existing `apnet3` floor,
and is recorded in model configuration.

**Legacy-parity bypass (established during implementation).** The three existing
induction-overlap call sites never applied a width floor, and flooring them is
*not* behavior-preserving: `AtomHirshfeldMPNN` emits `relu(...) + 1e-4`
(`ap2_hirshfeld_atom_model.py:403`), so sub-`0.1` widths are reachable, and on a
fixture containing widths of `0.04`/`0.07` an overlap edge worth ≈ -3.37 kcal/mol
collapses to ~0 under `clamp_min(0.1)`. Those three sites therefore pass
`width_floor=0.0`, which is an exact no-op for positive widths, while
`cliff_exchange` uses `OVERLAP_WIDTH_FLOOR`. The helper must accept `0.0` and the
divergence must be documented at each call site.

Refactor the three existing correct call sites (`mtp_mtp.py:2470-2476`,
`2592-2598`, `3051-3054`) to delegate to this helper. This is a pure
refactor: predictions must be bitwise-comparable within floating-point
tolerance, guarded by a regression test. `apnet3.py` is left alone.

### Exchange kernel

```python
def cliff_exchange(
    RA, RB,
    e_AB_source, e_AB_target,
    valence_widths_A, valence_widths_B,
    K_exch_A, K_exch_B,
    dR_AB: torch.Tensor | None = None,
    width_floor: float = OVERLAP_WIDTH_FLOOR,
) -> torch.Tensor:
```

Returns per-edge energy in kcal/mol:

\[
E_{ij}^{\mathrm{exch}} = K_i^{\mathrm{exch}} K_j^{\mathrm{exch}}\, S_{ij}
\times \texttt{h2kcalmol}
\]

Notes:

- `dR_AB` is an optional precomputed distance vector. When `None`, the kernel
  calls the existing `get_distances` (`mtp_mtp.py:1333`). Combined-mode
  forwards pass the distances already computed for electrostatics so the
  intermolecular distance reduction happens once per batch.
- The pair rule is the **product** \(K_i K_j\), not the geometric mean used by
  `geometric_mean_edge_values` (`mtp_mtp.py:1341`). The two rules must not be
  interchanged; positivity of \(K_i\) is supplied by the model's softplus
  contract, so no `abs` or clamping appears here.
- The result is strictly positive, matching the SAPT `Exch` sign convention.
- Exchange uses the full intermolecular edge domain (`e_ABfull_source`,
  `e_ABfull_target`), consistent with the Rackers routes.

### Generalized positive-parameter contract

`_validate_rackers_initialization` (`mtp_mtp.py:1186`) hard-codes four values.
Generalize it to:

```python
def _validate_positive_initialization(
    parameter_names, param_start_mean, param_start_std, positivity_epsilon
) -> tuple[list[float], list[float], float, list[float]]:
```

`_validate_rackers_initialization` becomes a thin wrapper binding
`RACKERS_PARAMETER_NAMES`, preserving its exact current error messages and
return shape. Error text for other contracts reports the expected count from
`len(parameter_names)`.

New module constants:

```python
CLIFF_EXCH_PARAMETER_NAMES = ("exch",)
CLIFF_EXCH_INITIAL_VALUES = (2.5,)
CLIFF_EXCH_INITIAL_STDS = (0.01,)

CLIFF_CLASSICAL_PARAMETER_NAMES = (
    "elst", "thole_direct", "thole_mutual", "ind_overlap", "exch",
)
CLIFF_CLASSICAL_INITIAL_VALUES = (1.8, 0.34, 0.39, 1.8, 2.5)
CLIFF_CLASSICAL_INITIAL_STDS = (0.01, 0.01, 0.01, 0.01, 0.01)

CLIFF_EXCH_INDEX = 0                 # within CliffExchangeNN output
CLIFF_CLASSICAL_ELST_INDEX = 0
CLIFF_CLASSICAL_THOLE_DIRECT_INDEX = 1
CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX = 2
CLIFF_CLASSICAL_IND_OVERLAP_INDEX = 3
CLIFF_CLASSICAL_EXCH_INDEX = 4
```

The `2.5` exchange default sits near the centre of CLIFF Table I
\(K^{\mathrm{exch}}\) values (0.60–7.60, mean ≈ 3.2) and reproduces
water-dimer-scale exchange at initialization.

Column 0–3 of `CLIFF_CLASSICAL_PARAMETER_NAMES` intentionally match the Rackers
ordering so the induction and electrostatics physics paths are reused unchanged
and a Rackers checkpoint's learned columns remain interpretable.

### `CliffExchangeNN` and `CliffClassicalNN`

Both subclass `AtomTypeParamNN` and follow the `RackersTholeDampingNN`
(`mtp_mtp.py:1263`) pattern exactly:

- reject nested models that are not `AtomTypeParamNN`,
- validate initialization through `_validate_positive_initialization`,
- initialize raw heads by inverse softplus so zeroed corrections map to the
  requested positive values,
- expose `K = softplus(K_raw) + positivity_epsilon` in `forward`, returning
  `(*output[:-1], parameters)` with `parameters` of shape
  `[n_atoms, len(parameter_names)]`,
- fix the output count; neither class accepts `n_params`,
- normalize the parameter tensor to two dimensions. `AtomTypeParamNN.forward`
  returns `K.squeeze(-1)` when `n_params == 1` (`mtp_mtp.py:1128`), so
  `CliffExchangeNN` must `unsqueeze(-1)` before applying softplus and return
  `[n_atoms, 1]`. Both new classes therefore present a uniform 2-D contract and
  `CLIFF_EXCH_INDEX` / `CLIFF_CLASSICAL_*_INDEX` column indexing works
  identically for either,
- emit `get_config()` carrying `model_type`, `parameter_names` in exact order,
  `param_start_mean`, `param_start_std`, `positivity_epsilon`, `width_floor`,
  readout settings, and `nested_atom_model`.

Valence widths continue to be read from the nested output as
`output[-2][:, 1]`, and Hirshfeld volume ratios as `abs(output[-2][:, 0])`.

### New `DimerProp` modes

Add four modes to `DimerProp.set_forward` (`mtp_mtp.py:136`), keeping the
existing dispatch style and updating its docstring:

| Mode | Model | Returns | Trains against |
| --- | --- | --- | --- |
| `cliff_exch` | `CliffExchangeNN` | `[n_edges]` | `y[:, 1]` |
| `cliff_classical` | `CliffClassicalNN` | `[n_edges, 3]` | `y[:, [0, 1, 2]]` |
| `cliff_classical_overlap` | `CliffClassicalNN` | `[n_edges, 3]` | `y[:, [0, 1, 2]]` |
| `cliff_classical_d3` | `CliffClassicalNN` | `[n_edges, 4]` | inference only |

Combined-column order is fixed at `(Elst, Exch, Indu)` and, for
`cliff_classical_d3`, `(Elst, Exch, Indu, Disp)` — matching the dataset's
`[Elst, Exch, Ind, Disp]` layout so target slicing is a plain column select.

`cliff_exch` requires no polarizability table. The three classical modes clone
`constants.polarizability_table` as the induction modes already do.

A shared `_cliff_classical_common_forward(batch, include_overlap, include_d3)`
mirrors `_rackers_thole_common_forward` (`mtp_mtp.py:208`) and:

1. evaluates the parameter model for `batch.batch_atomic_A` / `_B`,
2. reads columns via the named index constants, never negative magic indices,
3. computes intermolecular distances once and reuses them for exchange,
4. computes damped electrostatics with column 0 through the existing
   `mtp_elst_damping` / `mtp_elst_damping_AMOEBA` selection, passing
   `output[0].clone()` so induction's charge tensors are not mutated — the
   in-place hazard documented at `mtp_mtp.py:759`,
5. computes exchange with column 4,
6. computes induction with columns 1–3 via the existing
   `rackers_thole_induction`, forwarding `include_overlap`,
7. optionally computes `Disp = d3(batch, params=self.d3_damping_parameters)`,
8. returns `torch.vstack((...)).T` plus the monomer output tuples.

Ordering note: electrostatics must be evaluated on cloned charges, and the
existing `rackers_thole_induction` contract is unchanged.

### Training harnesses

Add to `mtp_mtp.py`, composing rather than duplicating `AM_DimerParam_Model`:

- `CliffExchangeModel` — `DIMER_EVAL = "cliff_exch"`,
  `MODEL_TYPE = "CliffExchangeNN"`, one parameter.
- `_CliffClassicalModelBase` with
  `CliffClassicalModel` (`DIMER_EVAL = "cliff_classical"`) and
  `CliffClassicalOverlapModel` (`DIMER_EVAL = "cliff_classical_overlap"`),
  `MODEL_TYPE = "CliffClassicalNN"`, five parameters.

These follow `_RackersTholeDampingModelBase` (`mtp_mtp.py:4918`), fixing model
type, dimer mode, parameter count, and ordering, and passing initialization
defaults through `_validate_positive_initialization`. As with the Rackers
routes, `train_models.py` forces `world_size = 1` for all CLIFF routes and
`AM_DimerParam_Model.train` continues to raise `NotImplementedError` for
`world_size > 1`; no DDP path is added.

Extend `_dimer_index_for_output` (`mtp_mtp.py:4302`), which currently switches on
the literal set `{"rackers_thole", "rackers_thole_overlap"}`, to also return
`batch.dimer_ind_full` for all four new modes. Replace the inline literal with a
module-level `FULL_EDGE_DIMER_EVAL_MODES` frozenset so the edge domain and the
aggregation index cannot drift apart. Every new mode uses `e_ABfull_*`, so all
four belong in that set; omitting one would silently scatter full-edge energies
with the short-range index.

Extend the `y_ind` / `term` dispatch at `mtp_mtp.py:4676` with:

- `cliff_exch` → `y_ind = 1`, `term = "Exch"`, and no polarizability or
  induction device moves,
- `cliff_classical` and `cliff_classical_overlap` → 
  `y_ind = torch.tensor([0, 1, 2])`, `term = "Elst      Exch      Ind"`,
  reusing the existing `AtomTypeParamNN` assertion and the same device
  placement block as the Rackers branch.

The existing tensor-`y_ind` path already handles multi-column MSE and per-column
reporting; extending it from two to three columns must not special-case width.

### Component/total loss weighting

Add an optional CLIFF Eq. (23) loss to the combined routes only:

\[
\mathcal{L} = (1-\gamma)\,\mathrm{MSE}(E_{\mathrm{total}})
+ \gamma \sum_{C} \mathrm{MSE}(E_C)
\]

- `component_gamma = None` is the default and selects the legacy plain
  multi-column MSE, bitwise unchanged, so existing runs are unaffected. Any
  float in `[0.0, 1.0]` selects the CLIFF Eq. (23) functional.

  **Why `None` rather than `1.0`.** An earlier draft made `1.0` the default and
  claimed it reproduced legacy behavior. It cannot: `sum_C MSE(E_C)` is `k` times
  the plain mean MSE over `k` columns, so overloading `1.0` to mean both produced
  a measured `3x` loss discontinuity between `gamma = 0.9999` (6.802) and
  `gamma = 1.0` (2.267) at `k = 3`. That jumps the effective learning rate
  mid-sweep, and sweeping gamma from 0 to 1 is exactly the CLIFF Fig. 3 analysis
  this feature exists to support. Decoupling "which functional" from "what gamma"
  removes the discontinuity without renormalizing, so `component_gamma = 0.4`
  still means the paper's 0.4. The component term is deliberately left
  UNNORMALIZED to preserve that fidelity; `gamma = 1.0` is a legitimate CLIFF
  endpoint meaning "fit purely to component energies".
- When `component_gamma` is not `None`, `E_total` defaults to the **partial** total
  `Elst + Exch + Indu` compared against `y[:, 0] + y[:, 1] + y[:, 2]`. This
  keeps the total term differentiable with respect to only trained parameters.
- `total_includes_d3 = False` by default. When enabled, `E_total` adds
  `d3(batch)` and is compared against all four SAPT columns, matching CLIFF's
  full-total fit at the cost of a D3 evaluation per batch. D3 contributes no
  gradient.
- The reference total is always formed by **summing** SAPT component columns.
  `util.load_dimer_dataset` can supply a `Total_aug` column
  (`util.py:196-231`), but `ap2_fused_ds` deliberately stores only the four
  components (`ap2_fused_ds.py:1201`). Summing avoids a dataset schema change,
  which is an explicit non-goal.
- CLIFF's fitted value is `gamma = 0.4`; it is exposed as a CLI option, not
  hard-coded as the default.
- `train_models.py` already defines an `--include_total_mse` flag, which is
  currently filtered out for `AM_DimerParam_Model.train` because that method does
  not accept it. `component_gamma` supersedes it for the CLIFF routes:
  `--include_total_mse` on a CLIFF route is interpreted as
  `component_gamma = 0.5` unless `--component_gamma` is given explicitly, and
  passing both raises `ValueError`. Behavior for every other route is unchanged.
- The shared pairwise tail filters `train_kwargs` through
  `inspect.signature(apnet.train).parameters` (`train_models.py:694-728`), so the
  new harnesses must accept `component_gamma` and `total_includes_d3` as named
  `train()` parameters for them to be forwarded at all.
- `component_gamma` outside `[0.0, 1.0]` raises `ValueError`. Any non-`None`
  value on `CliffExchangeModel` raises `ValueError`, since a single-component
  route has no meaningful total/component split. `total_includes_d3 = True` with
  `component_gamma is None` also raises, since the total would carry zero weight
  and the flag would be a silent no-op.

### Two-stage fitting route

CLIFF fits components individually, then refits jointly. Support this through
existing checkpoint plumbing rather than new orchestration:

1. Train `RackersTholeDampingModel` (elst + induction) and `CliffExchangeModel`
   (exch) independently.
2. Warm-start `CliffClassicalModel` from those checkpoints via
   `pre_trained_model_path`, mapping Rackers columns 0–3 and exchange column 0
   into classical columns 0–4.

Add a focused helper for step 2:

```python
def merge_classical_parameter_checkpoints(
    rackers_checkpoint_path: str | None,
    exchange_checkpoint_path: str | None,
    output_path: str,
) -> dict:
```

Because per-parameter state lives in parallel `nn.ModuleList`s indexed by
parameter position — `guess_layer[p]` (a `[max_Z + 1, 1]` embedding) and
`param_readout_layers[p]` (its `n_message + 1` MLPs) — remapping a column is a
module-index remap of those two lists. The helper:

- validates that both checkpoints declare compatible nested `AtomTypeParamNN`
  configurations and identical `n_message` / `n_neuron` / `n_embed`,
- reads each source checkpoint's `parameter_names` and copies
  `guess_layer.{p}` and `param_readout_layers.{p}.*` state into the destination
  index for that name, so remapping is driven by metadata and never by
  positional assumption,
- leaves unclaimed destination columns at their initialization values,
- raises `ValueError` on parameter-name, architecture, or tensor-shape mismatch.

### `cliff_2.py`

New file `src/apnet_pt/AtomPairwiseModels/cliff_2.py` exporting `CLIFF2Model`.

Responsibilities:

- Load a trained classical parameter set and expose full SAPT0 prediction.
- Accept either a single `CliffClassicalNN` checkpoint
  (`classical_model_path`) or separate component checkpoints
  (`rackers_model_path` + `exchange_model_path`), the latter routed through
  `merge_classical_parameter_checkpoints` into an in-memory merged model.
  Supplying both forms, or neither, raises `ValueError`.
- Own dispersion via `resolve_d3_damping_parameters` / `d3`, with an optional
  `d3_damping_parameters` override, defaulting to the value recorded in the
  loaded checkpoint config.
- Select `cliff_classical_d3` as its `DimerProp` mode, with `include_overlap`
  read from checkpoint config rather than re-specified by the caller.

Public surface:

- `forward(batch) -> torch.Tensor` of shape `[n_edges, 4]`,
  columns `(Elst, Exch, Indu, Disp)` in kcal/mol.
- `predict_batch(batch) -> torch.Tensor` of shape `[n_dimers, 5]`, the
  scatter-aggregated components plus a trailing total column. Aggregation uses
  the existing `scatter_sum_compile` and `_dimer_index_for_output`
  (`mtp_mtp.py:4302`) conventions so dimer indexing matches the training path.
- `predict_qcel_mols_dimer(mols, batch_size=...)` returning per-dimer
  components and total, mirroring `AM_DimerParam_Model.predict_qcel_mols_dimer`
  (`mtp_mtp.py:4311`) argument and return conventions.
- `component_labels` → `("Elst", "Exch", "Indu", "Disp", "Total")`.
- `info()` delegating to `model_print.model_tree_string`, as `DimerProp.info`
  does.

Inference posture:

- The constructor calls `eval()` and `requires_grad_(False)` on the whole
  hierarchy.
- Prediction entry points wrap work in `torch.inference_mode()`.
- No training, optimizer, dataset-construction, or DDP code lives in this file.
- Register `CLIFF2Model` in `AtomPairwiseModels/__init__.py` alongside the
  existing exports.

## Checkpoint contract

New checkpoints use the existing `model_io` v-current format. Saved config adds:

- `model_type` — `"CliffExchangeNN"` or `"CliffClassicalNN"`,
- `parameter_names` in exact order,
- `param_start_mean`, `param_start_std`, `positivity_epsilon`,
- `width_floor`,
- `dimer_eval` — one of the four new modes,
- `elst_damping_type`,
- `d3_damping_parameters`,
- `component_gamma` and `total_includes_d3` for combined routes,
- `nested_atom_model` config sufficient for standalone reconstruction.

`AM_DimerParam_Model.__init__` currently hard-codes its pre-trained checkpoint
validation to `model_type == "RackersTholeDampingNN"` and
`config["parameter_names"] == list(RACKERS_PARAMETER_NAMES)`
(`mtp_mtp.py:3655-3701`). Generalize that block to look the expected contract up
from a `{model_type: parameter_names}` mapping covering `RackersTholeDampingNN`,
`CliffExchangeNN`, and `CliffClassicalNN`, preserving the existing
`checkpoint_version`, `model_io.validate_checkpoint`, `dimer_eval`-agreement, and
`nested_atom_model` rebuild checks verbatim. The Rackers error messages must not
change.

Loading validates that `parameter_names` is present and exactly equal to the
contract for the declared `model_type`. A missing, reordered, or unknown
parameter list raises `ValueError` rather than silently reassigning physical
meaning — the same guarantee the Rackers loader provides. Extend
`load_dimer_prop_from_checkpoint` (`mtp_mtp.py:868`) and
`_infer_atomtypeparamnn_from_state_dict` (`mtp_mtp.py:827`) to recognize the
new model types; existing non-CLIFF reconstruction paths stay unchanged.

`CLIFF2Model` checkpoints record the source classical checkpoint config plus
resolved D3 parameters, so a `CLIFF2Model` reloads without its constituent
files.

## Training route

Extend `RACKERS_MODEL_TYPES`-style dispatch in `train_models.py` (currently
lines 14-15, 369-372) with the exact identifiers `CliffExchangeModel`,
`CliffClassicalModel`, and `CliffClassicalOverlapModel`, and update
`--train_apnet` help text.

Canonical standalone exchange run:

```bash
python train_models.py \
  --train_apnet CliffExchangeModel \
  --am_model_path <multipole-model.pt> \
  --atom_type_param_model_path <hfvr-vw-model.pt> \
  --ap_model_path <cliff-exch.pt> \
  --data_dir <pairwise-dataset>
```

Canonical joint classical run with CLIFF's fitted weighting:

```bash
python train_models.py \
  --train_apnet CliffClassicalOverlapModel \
  --am_model_path <multipole-model.pt> \
  --atom_type_param_model_path <hfvr-vw-model.pt> \
  --ap_model_path <cliff-classical.pt> \
  --data_dir <pairwise-dataset> \
  --component_gamma 0.4
```

CLI details:

- Add `--component_gamma` (float or unset, default unset/`None`) and `--total_includes_d3`
  (flag). Both are rejected for `CliffExchangeModel` and for all pre-existing
  routes.
- Reuse the existing unset-sentinel treatment of `--param_start_mean` and
  `--param_start_std`: resolve to the CLIFF defaults only on CLIFF routes,
  preserve current defaults elsewhere. Overrides require exactly one value for
  the exchange route and exactly five for the classical routes; scalar
  broadcasting is rejected so parameter intent stays unambiguous.
- `--n_params` must not configure either new model.
- `--unfreeze_atom_model` continues to control nested fine-tuning; the nested
  model is frozen by default with `requires_grad_(False)`.

Also add a `merge_classical_parameter_checkpoints` entry point so stage-two
warm starts are reproducible from the CLI.

## Error handling

Raise clear `ValueError`s for:

- initialization lists whose length does not match the declared
  `parameter_names`,
- `component_gamma` outside `[0.0, 1.0]`, or any non-`None` value supplied to a
  single-component route,
- `total_includes_d3 = True` while `component_gamma is None`,
- a model-config `width_floor` that is `<= 0` or non-finite (the
  `atomic_overlap_S_ij` argument itself accepts `0.0` as a documented
  legacy-parity bypass; see the shared-helper section),
- unsupported nested atom-model types,
- `cliff_exchange` receiving mismatched edge-index and distance lengths,
- checkpoints with absent, reordered, or unknown `parameter_names`,
- `CLIFF2Model` receiving both or neither of the single-checkpoint and
  component-checkpoint forms,
- `merge_classical_parameter_checkpoints` given incompatible nested atom-model
  configs or parameter-name metadata,
- unknown `dimer_eval` values, through the existing dispatcher.

Positivity of every predicted \(K\) is guaranteed by softplus. Downstream code
must not use `abs` or clamping to conceal a violation of that contract. The one
sanctioned clamp is `width_floor` on valence widths inside
`atomic_overlap_S_ij`, which guards the `rsqrt` against a degenerate predicted
width.

## Testing strategy

Add `tests/test_cliff_classical_exchange.py` and `tests/test_cliff_2.py`. All
tests are CPU-only and require no network or database access.

### Golden overlap value test

Assert `atomic_overlap_S_ij` against the hand-derived reference from the
verification section: with `sigma_A = 0.39`, `sigma_B = 0.36`, `r = 1.95` Å
expressed in bohr, `B_ij == pytest.approx(2.6688, rel=1e-4)` and
`S_ij == pytest.approx(2.30760e-3, rel=1e-4)`. Then assert
`cliff_exchange` with `K_A = 5.8538`, `K_B = 0.5996` returns
`pytest.approx(5.0825, rel=1e-3)` kcal/mol. This pins both the square-root
form of \(B_{ij}\) and the `h2kcalmol` placement.

Additionally assert that the literal \(1/(\sigma_i\sigma_j)\) form is *not* what
ships, by checking the returned value differs from the `apnet3` convention by
more than six orders of magnitude.

### Overlap helper property tests

- symmetry under exchanging `(A, source)` with `(B, target)`,
- monotonic decay in `r` and in `B_ij`,
- `S_ij → 1` as `r → 0`,
- dtype and device preservation,
- `width_floor` engagement for a predicted width below the floor,
- no `[n_A, n_B]` intermediate is materialized, verified by asserting output
  shape equals edge count for a case where `n_A * n_B` differs from it.

### Refactor equivalence test

For randomized inputs, assert the refactored induction-overlap call sites
(`rackers_thole_induction` with `include_overlap=True`,
`induced_dipole_induction`, `induced_dipole_induction_optimized`) reproduce
pre-refactor values within floating-point tolerance.

### `apnet3` legacy pin test

Assert `apnet3.valence_width_exch` still returns its current values for a fixed
input, documenting in the test body that the non-physical \(B_{ij}\) is
absorbed by `readout_layer_exch_quotient` and must not be "corrected" without
retraining AP3 checkpoints.

### Parameter-head tests

For `CliffExchangeNN` and `CliffClassicalNN`, with a deterministic nested
atom-model fixture:

- output shapes `[n_atoms, 1]` and `[n_atoms, 5]`,
- outputs finite and strictly positive,
- zeroed correction heads initialize near `(2.5,)` and
  `(1.8, 0.34, 0.39, 1.8, 2.5)`,
- wrapped multipole and auxiliary outputs preserved,
- finite gradients at every readout head,
- `n_params` is not accepted.

### Combination-rule test

Verify exchange uses the product \(K_i K_j\) and not the geometric mean, by
constructing values where the two differ and asserting the product result.
Verify `geometric_mean_edge_values` is not called on the exchange path.

### Physics-routing test

With controlled parameters or monkeypatched kernels, verify for each new mode:

- column 0 reaches only electrostatic damping,
- columns 1 and 2 reach only direct and mutual Thole damping respectively,
- column 3 is used only by the overlap route,
- column 4 reaches only `cliff_exchange`,
- `cliff_exch` never constructs a polarizability table or runs induction,
- exchange and electrostatics consume the same `e_ABfull_*` edge domain,
- intermolecular distances are computed once per forward pass.

### Sign and magnitude test

On a real close-contact dimer fixture from `tests/test_data_path/test_geoms`
(for example `mol_cliff_water_close.dat`), assert predicted exchange is
strictly positive for every edge and that dimer-level exchange decays
monotonically as the monomers are separated.

### Joint-forward and gradient test

On a small synthetic CPU dimer batch:

- `cliff_exch` returns finite `[n_edges]`,
- `cliff_classical` and `cliff_classical_overlap` return finite
  `[n_edges, 3]`,
- `cliff_classical_d3` returns finite `[n_edges, 4]`,
- scatter aggregation yields finite `[batch_size, k]`,
- backward reaches all energy-active heads (four for `cliff_classical`, five
  for the overlap route, one for `cliff_exch`),
- parameters remain positive after one optimizer step.

### Loss-weighting test

- `component_gamma = None` reproduces the plain multi-column MSE bitwise,
- `component_gamma = 0.0` yields a loss depending only on the summed total,
- `component_gamma = 0.4` matches a hand-computed value,
- the CLIFF branch is continuous across the sweep: `gamma = 1.0` and
  `gamma = 0.9999` agree to tight tolerance, and `gamma = 1.0` equals
  `sum_C MSE(E_C)` exactly,
- `total_includes_d3 = True` changes the total but contributes no D3 gradient,
- out-of-range gamma and gamma on `CliffExchangeModel` raise `ValueError`.

### Training-dispatch test

Monkeypatching heavyweight dataset/model construction, verify
`train_pairwise_model` for each of the three new identifiers:

- selects the intended harness and dimer mode,
- supplies the correct default initialization tuple,
- sets `y_ind` to `1` for exchange and `[0, 1, 2]` for the classical routes,
- forwards multipole and HFVR/valence-width checkpoint paths,
- freezes the nested atom model by default and honors `--unfreeze_atom_model`,
- rejects scalar broadcasting and wrong-length parameter overrides.

### Checkpoint round-trip tests

For each new model type, save and reload, then verify predictions match within
tolerance, parameter metadata and ordering are preserved, nested state is
restored without a separately supplied object, and a reordered
`parameter_names` config is rejected.

### Merge helper test

Verify `merge_classical_parameter_checkpoints` maps Rackers columns 0–3 and
exchange column 0 into classical columns 0–4 by name, leaves unclaimed columns
at initialization values, and raises on nested-config or parameter-name
mismatch.

### `CLIFF2Model` tests

- construction from a single classical checkpoint and from component
  checkpoints produce identical predictions,
- both-forms and neither-form constructor arguments raise `ValueError`,
- `forward` returns `[n_edges, 4]`; `predict_batch` returns `[n_dimers, 5]`
  with the total column equal to the sum of the four components,
- `include_overlap` is taken from checkpoint config, not caller input,
- the model is in `eval()` mode with no parameter requiring grad,
- `d3_damping_parameters` override changes only the dispersion column,
- `predict_qcel_mols_dimer` runs on a two-dimer geometry fixture and returns
  finite components,
- a round-trip save/load reproduces predictions.

## Verification commands

```bash
python -m pytest tests/test_cliff_classical_exchange.py -v
python -m pytest tests/test_cliff_2.py -v
python -m pytest tests/test_rackers_thole_damping.py -v
python -m pytest tests/test_polarization.py -v
python -m pytest tests/test_model_io.py -v
python -m pytest tests/test_atomtype_props.py -v
python train_models.py --help
python -m pytest tests/ -m "not slow" -q
```

`tests/test_rackers_thole_damping.py` and `tests/test_polarization.py` are the
regression gates for the shared-helper refactor. There is no `tests/test_apnet3.py`
in this repository, so the `apnet3` legacy-pin test is added to
`tests/test_cliff_classical_exchange.py` rather than to an apnet3-specific file.

Reuse the existing fixtures in `tests/test_rackers_thole_damping.py`
(`synthetic_dimer_batch`, `atomic_batch`, `synthetic_qcel_dimers`,
`nested_hfvr_vw_model` at `:153-237`) and its monkeypatch fakes
(`_FakeHFVRModel`, `_FakeAtomTypeParamModel`, `_FakeRackersHarnessBase` at
`:2251-2315`) rather than duplicating them; promote them to a shared conftest
fixture module if both test files need them.

## Amendment: trainability of the positive-parameter heads (2026-08-21)

The design as originally specified is functionally correct and passes its whole
test suite, but the first 100-epoch run on a 5000-dimer subset did not learn.
This section records what failed, why, and the four changes made in response, so
that the initialization contract above is read together with its correction.

### What the first run did

Exchange converged to a train MAE of 6.803, which is *exactly* the mean absolute
`Exch` of the training subset — the predict-zero MAE. A held-out evaluation over
640 dimers gave Pearson r = 0.0139 and R^2 = -0.1817, with predictions spanning
`[0.00, 14.29]` against a reference span of `[0.00, 168.03]`. Dumping the trained
`CliffClassicalNN` over 4264 atoms from 160 test dimers showed the cause was not
confined to exchange:

| column | seed | mean | median | max | % at `positivity_epsilon` |
|---|---:|---:|---:|---:|---:|
| `elst` | 1.8 | 22.17 | 13.86 | 164.7 | 0% |
| `thole_direct` | 0.34 | 0.105 | 0.057 | 2.31 | 0% |
| `thole_mutual` | 0.39 | 0.123 | 0.033 | 0.68 | 18% |
| `ind_overlap` | 1.8 | 0.028 | 0.012 | 1.79 | 24% |
| `exch` | 2.5 | 0.025 | 0.008 | 2.49 | 23% |

Because `ind_overlap` was among the dead columns, `cliff_classical` and
`cliff_classical_overlap` became numerically indistinguishable (val Elst 2.846
vs 2.847, Exch 8.079 vs 8.080, Ind 2.605 vs 2.573), so the controlled comparison
that motivates having both routes could not be made at all.

The physics was cleared first and is unchanged: predicted valence widths match
CLIFF Fig. 5 to 2-4% (H 0.373 / C 0.521 / N 0.443 / O 0.406 against 0.36 / 0.50 /
0.45 / 0.39), and the Eq. (11) kernel reproduces its golden values.

### Root causes

1. **Softplus collapse is irreversible.** `K = softplus(raw) + epsilon` has
   `dK/draw = sigmoid(raw)`. A column driven toward zero loses the gradient that
   would bring it back: at `raw = -30` the derivative is 9.4e-14. Whatever
   pushed a column down early in training permanently disabled it.
2. **A uniform `K_exch` is wrong in the one place it matters most.** Eq. (8)
   combines the parameter multiplicatively, so a uniform 2.5 makes an H-H pair
   `6.25` where CLIFF's hydrogen types give `0.77 * 0.77 = 0.59`. Hydrogen
   dominates the atom counts, so the cheapest way for the optimizer to remove
   that error is to shrink every `K_exch` at once — straight into (1).
3. **The random readout swamped the seed.** Measured at initialization, the
   randomly drawn correction MLP contributed more exchange error than the entire
   uniform-versus-per-element difference, so any care taken over seeds was
   largely wasted.
4. **Nothing bounded the update.** Gradient clipping was present but commented
   out, SAPT components reach ~240 kcal/mol under MSE, and at `lr = 5e-4` Adam's
   cumulative displacement over 100 epochs is by itself enough to carry a raw
   parameter from its seed into saturation.

### Changes

- **`CLIFF_EXCH_INITIAL_VALUES_BY_Z`** seeds `K_exch` per element from CLIFF
  Table I (H 0.77, C 2.40, N 4.20, O 5.60, F 7.60, S 3.20, Cl 3.80, Br 4.10).
  Elements absent from the table fall back to the scalar seed. Columns 0-3 of
  the combined head keep their scalar Rackers seeds; only `exch` has published
  per-element values.
- **`CLIFF_PARAM_FLOOR_FRACTION = 0.05` / `CLIFF_PARAM_CEILING_MULTIPLE = 10.0`**
  clamp each raw parameter to `[0.05x, 10x]` of its column's seed through
  `_ste_clamp`, which bounds the value on a detached copy and reattaches an
  identity gradient. A collapsing column therefore parks where `sigmoid(raw)` is
  still order 0.1 rather than 1e-8, and stays able to climb back out; the ceiling
  catches the `elst` runaway. Both are config-recorded and both accept `None`
  to reproduce the pre-bound forward exactly, which is what a checkpoint
  predating them loads as.
- **`CLIFF_READOUT_INIT_SCALE = 0.1`** scales the *output* layer of each readout
  MLP at construction, so the per-element seed governs the initial prediction.
  Only the output layer is scaled, which keeps the knob linear in the emitted
  correction; scaling the whole four-deep stack would compound as `s ** 4`.
- **`grad_clip_norm`** is a real, opt-in `train()` parameter and
  `--grad_clip_norm` CLI flag, defaulting to `None` so every pre-existing route
  is bitwise unchanged. `run_cliff.sh` sets it to 1.0 and drops the default
  learning rate to 1e-4.

At-init exchange MAE against SAPT `Exch` on held-out dimers, predict-zero
baseline 18.555:

| configuration | MAE |
|---|---:|
| uniform 2.5, no bounds, full readout | 7.04 |
| + per-element seeds | 3.37 |
| + readout x 0.1 (shipped default) | 3.10 |

### Notes for the next reader

- `None` is a *meaningful* value for all four knobs (no per-element table, no
  bound, no readout scaling), so "unspecified" needs the distinct
  `_CLIFF_HEAD_DEFAULT` sentinel. Passing `None` through as a default would
  silently disable each feature for every caller that did not name it.
- The bounds are registered as **non-persistent** buffers. They are fully
  determined by config, so keeping them out of `state_dict` avoids a second
  source of truth and leaves checkpoints loadable by builds predating them.
- The obvious way to write a straight-through clamp,
  `x + (bound - x).detach()`, is algebraically right but loses the bound to
  catastrophic cancellation once `|x|` is large: a readout at `raw = 3e7` came
  out at 32 rather than the requested 25. Clamp a detached copy instead.

## Incidental findings (documented, not addressed here)

These were found while mapping the existing code. None is in scope; each is
recorded so the next reader does not rediscover it.

- `DimerProp.set_forward` maps `dimer_eval="elst_damping_AMOEBA"` to
  `self._elst_damping_AMOEBA_forward`, but the defined method is
  `_elst_damping_forward_AMOEBA` (`mtp_mtp.py:369`). That standalone mode raises
  `AttributeError`. The AMOEBA path used by the Rackers and CLIFF combined
  forwards is a different, working call into `mtp_elst_damping_AMOEBA` and is
  unaffected.
- `apnet3.py:307` defines an unfinished `induced_dipole_indu(..., S_ij, ...)`
  stub with debug prints — a partial CLIFF port that nothing calls.
- `mtp_elst` mutates `qA` / `qB` in place (`mtp_mtp.py:1671-1672`), which is why
  the combined forwards must clone charges.
- `rackers_thole_induction` accepts `quadA` / `quadB` and immediately deletes
  them (`mtp_mtp.py:2276`); quadrupole induction is not implemented.

## Acceptance criteria

The feature is accepted when:

1. `atomic_overlap_S_ij` implements \(B_{ij} = 1/\sqrt{\sigma_i \sigma_j}\),
   returns a dimensionless overlap, and reproduces the golden water-dimer
   values.
2. All three previously correct induction-overlap call sites delegate to the
   shared helper with no change in predictions, and `apnet3` numerics are
   unchanged and pinned by test.
3. `cliff_exchange` returns strictly positive per-edge energies in kcal/mol
   using the multiplicative \(K_i K_j\) rule over the full AB edge domain, and
   reuses caller-supplied distances when given.
4. `CliffExchangeNN` and `CliffClassicalNN` return positive, correctly
   initialized per-atom parameters with stable documented column meanings.
5. Each predicted column reaches exactly its intended physics term, and
   intermolecular distances are computed once per forward pass.
6. All four new modes are members of `FULL_EDGE_DIMER_EVAL_MODES` and aggregate
   with `batch.dimer_ind_full`, and `CliffExchangeNN` returns `[n_atoms, 1]`
   despite the `n_params == 1` squeeze in `AtomTypeParamNN.forward`.
7. `--train_apnet CliffExchangeModel` trains exchange alone against SAPT
   `Exch`, and both classical identifiers train elst + exch + induction
   jointly, freezing the nested atom model by default.
8. `component_gamma` defaults to `None` and reproduces existing loss behavior
   bitwise; `0.4` reproduces CLIFF's Eq. (23) weighting; and the CLIFF branch is
   continuous across the whole gamma sweep, with `gamma = 1.0` equal to
   `sum_C MSE(E_C)`.
9. `merge_classical_parameter_checkpoints` enables a two-stage
   individual-then-joint fit by name-driven column remapping.
10. `CLIFF2Model` in `cliff_2.py` produces four finite SAPT0 components plus a
    total from either a combined or component checkpoint set, is inference-only
    and gradient-free, and round-trips through `model_io`.
11. Checkpoint validation is contract-driven for all three parameter models,
    with Rackers error messages unchanged.
12. All new focused tests pass and existing Rackers, polarization, model_io,
    and atom-type regressions are unchanged.
13. (Amendment) No parameter column can be driven to a state with no usable
    gradient: a hostile readout parks each column at its configured floor with
    `dK/draw` still order 0.1, and a runaway readout is capped at the ceiling.
    `K_exch` is seeded per element, the readout correction is scaled so that
    seed governs the initial prediction, and `grad_clip_norm` defaults to `None`
    so every pre-existing route is bitwise unchanged.
