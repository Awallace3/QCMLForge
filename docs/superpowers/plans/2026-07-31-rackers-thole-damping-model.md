# Rackers Thole Damping Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four-head Rackers Thole damping with pure-IPD and overlap-enabled harnesses, full-edge training/prediction, compatible standalone checkpoints, and CLI dispatch for both variants.

## Architecture

`RackersTholeDampingNN` subclasses the existing atom-parameter readout pattern while fixing four positive output columns. `DimerProp` routes electrostatic, direct-Thole, mutual-Thole, and overlap columns into a shared Rackers induction kernel. Two thin harness subclasses select pure-IPD or overlap-enabled induction while reusing `AM_DimerParam_Model` training, prediction, and checkpoint infrastructure.

## Tech Stack

Python, PyTorch, PyTorch Geometric, qcelemental, pytest, repository checkpoint-v2 utilities.

## Global Constraints

Copied from the approved design:

- Predict distinct electrostatic, direct-Thole, mutual-Thole, and induction overlap parameters for every atom.
- Guarantee that all four predicted parameters are positive; the two Thole parameters must be suitable for square-root combination.
- Support controlled performance comparison between pure induced-point-dipole induction and induction augmented with a learned short-range overlap term.
- Apply direct and mutual Thole values to their physically distinct interaction tensors.
- Train the model jointly against SAPT electrostatics and induction components.
- Preserve all existing `AtomTypeParamNN`, `AM_DimerParam_Model`, and checkpoint behavior.
- Provide focused, CPU-compatible pytest coverage for model behavior, physics routing, training dispatch, and checkpoint round trips.
- Do not introduce categorical atom-type labels or classification losses.
- Do not predict parameters directly on atom-pair edges.
- Do not change the output semantics of the existing `AtomTypeParamNN`.
- Do not migrate existing checkpoints to the new model.
- Do not alter the pairwise dataset schema or SAPT component ordering.
- Do not introduce component loss weights; electrostatics and induction retain equal MSE weighting.
- Do not refactor unrelated electrostatic or polarization implementations.
- The nested pretrained `AtomTypeParamNN` supplies both Hirshfeld volume ratios and valence widths; both Rackers variants use HFVR for polarizability scaling, while only the overlap variant uses valence widths in its energy expression.
- The nested model is frozen by default. Freezing must be enforced with `requires_grad_(False)`.
- `RackersTholeDampingNN` owns exactly four heads ordered as `("elst", "thole_direct", "thole_mutual", "ind_overlap")`; it must not accept a variable `n_params`.
- Public parameters use `softplus(raw) + 1e-8`, initialized through inverse softplus from `[1.8, 0.34, 0.39, 1.8]`.
- The raw initialization standard deviations default to `[0.01, 0.01, 0.01, 0.01]`.
- Direct and mutual edge parameters use `sqrt(K_i * K_j)` independently for AB, AA, and BB edges.
- `rackers_thole` excludes overlap energy but retains column 3 for downstream compatibility.
- `rackers_thole_overlap` subtracts the existing `K_A * S_ij * K_B` overlap contribution using column 3.
- Training and prediction use `e_ABfull_source`, `e_ABfull_target`, and `dimer_ind_full`.
- Both harnesses train against SAPT columns `[0, 2]` with equal MSE weighting.
- Rackers checkpoint loading must reject missing or reordered parameter metadata.
- Rackers initialization overrides must contain exactly four comma-separated values; scalar broadcasting remains available only to existing routes.
- Downstream Rackers code must not use `abs` or clamping to conceal violations of the positivity contract.

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `src/apnet_pt/pt_datasets/ap2_fused_ds.py` | Modify | Add full AB edges and `dimer_ind_full` to target-bearing collation without changing stored records. |
| `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py` | Modify | Add constants, geometric combination helper, four-head model, Rackers induction kernel, two `DimerProp` modes, harnesses, full-edge training aggregation, and checkpoint reconstruction. |
| `train_models.py` | Modify | Add both training identifiers, route-specific defaults and validation, HFVR/VW construction, freeze forwarding, and help text. |
| `tests/test_rackers_thole_damping.py` | Create | Focused CPU-only tests for all new contracts. |
| `tests/test_polarization.py` | Modify | Add named direct/mutual Thole tensor regression so the required `-k thole` command selects a real test. |

## Task Interfaces

Implement these exact public constants and signatures in `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`:

```python
RACKERS_PARAMETER_NAMES = (
    "elst",
    "thole_direct",
    "thole_mutual",
    "ind_overlap",
)
RACKERS_INITIAL_VALUES = (1.8, 0.34, 0.39, 1.8)
RACKERS_INITIAL_STDS = (0.01, 0.01, 0.01, 0.01)
RACKERS_POSITIVITY_EPSILON = 1e-8

RACKERS_ELST_INDEX = 0
RACKERS_THOLE_DIRECT_INDEX = 1
RACKERS_THOLE_MUTUAL_INDEX = 2
RACKERS_IND_OVERLAP_INDEX = 3
```

```python
def geometric_mean_edge_values(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    e_source: torch.Tensor,
    e_target: torch.Tensor,
) -> torch.Tensor:
```

```python
class RackersTholeDampingNN(AtomTypeParamNN):
    def __init__(
        self,
        atom_model: AtomTypeParamNN,
        n_message: int = 3,
        n_neuron: int = 128,
        n_embed: int = 8,
        param_start_mean: tuple[float, float, float, float] = RACKERS_INITIAL_VALUES,
        param_start_std: tuple[float, float, float, float] = RACKERS_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        freeze_atom_model: bool = True,
    ):
```

```python
def rackers_thole_induction(
    ZA: torch.Tensor,
    RA: torch.Tensor,
    qA: torch.Tensor,
    muA: torch.Tensor,
    quadA: torch.Tensor,
    ZB: torch.Tensor,
    RB: torch.Tensor,
    qB: torch.Tensor,
    muB: torch.Tensor,
    quadB: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    e_AA_source: torch.Tensor,
    e_BB_source: torch.Tensor,
    e_AA_target: torch.Tensor,
    e_BB_target: torch.Tensor,
    hirshfeld_volume_ratio_A: torch.Tensor,
    hirshfeld_volume_ratio_B: torch.Tensor,
    valence_widths_A: torch.Tensor,
    valence_widths_B: torch.Tensor,
    thole_direct_A: torch.Tensor,
    thole_direct_B: torch.Tensor,
    thole_mutual_A: torch.Tensor,
    thole_mutual_B: torch.Tensor,
    ind_overlap_A: torch.Tensor,
    ind_overlap_B: torch.Tensor,
    include_overlap: bool = False,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    polarizability_table: torch.Tensor = constants.polarizability_table,
) -> torch.Tensor:
```

```python
class RackersTholeDampingModel(AM_DimerParam_Model):
    DIMER_EVAL = "rackers_thole"
```

```python
class RackersTholeDampingOverlapModel(AM_DimerParam_Model):
    DIMER_EVAL = "rackers_thole_overlap"
```

The harness constructors must share a private base implementation and expose the existing `AM_DimerParam_Model` dataset, training, prediction, save, and load arguments. Neither public constructor may expose `n_params`, `model_type`, or `dimer_eval_type`.

---

### Task 1: Establish the target-bearing full-edge batch contract

**Files:**
- Create: `tests/test_rackers_thole_damping.py`
- Modify: `src/apnet_pt/pt_datasets/ap2_fused_ds.py`

- [ ] **Step 1: Write the failing collation test**

Add a small `Data` factory and test:

```python
def _make_collate_item(y_scale: float) -> Data:
    return Data(
        y=torch.tensor(
            [-1.0, 2.0, -3.0, 4.0], dtype=torch.float32
        ) * y_scale,
        ZA=torch.tensor([8, 1], dtype=torch.long),
        RA=torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        ZB=torch.tensor([8, 1], dtype=torch.long),
        RB=torch.tensor(
            [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        e_ABsr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABsr_target=torch.tensor([0, 0], dtype=torch.long),
        e_ABlr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABlr_target=torch.tensor([1, 1], dtype=torch.long),
        e_AA_source=torch.tensor([0, 1], dtype=torch.long),
        e_AA_target=torch.tensor([1, 0], dtype=torch.long),
        e_BB_source=torch.tensor([0, 1], dtype=torch.long),
        e_BB_target=torch.tensor([1, 0], dtype=torch.long),
        dimer_ind=torch.zeros(2, dtype=torch.long),
        dimer_ind_lr=torch.zeros(2, dtype=torch.long),
        molecule_ind_A=torch.zeros(2, dtype=torch.long),
        molecule_ind_B=torch.zeros(2, dtype=torch.long),
        total_charge_A=torch.tensor(0.0),
        total_charge_B=torch.tensor(0.0),
    )


def test_target_collate_emits_full_edge_domain():
    batch = ap2_fused_collate_update(
        [_make_collate_item(1.0), _make_collate_item(2.0)]
    )

    assert torch.equal(
        batch.e_ABfull_source,
        torch.cat((batch.e_ABsr_source, batch.e_ABlr_source)),
    )
    assert torch.equal(
        batch.e_ABfull_target,
        torch.cat((batch.e_ABsr_target, batch.e_ABlr_target)),
    )
    assert torch.equal(
        batch.dimer_ind_full,
        torch.cat((batch.dimer_ind, batch.dimer_ind_lr)),
    )
    assert batch.e_ABfull_source.numel() == batch.dimer_ind_full.numel()
    assert batch.dimer_ind_full.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tests/test_rackers_thole_damping.py::test_target_collate_emits_full_edge_domain -v
```

Expected: failure because `ap2_fused_collate_update` lacks `e_ABfull_source`, `e_ABfull_target`, and `dimer_ind_full`.

- [ ] **Step 3: Implement full-edge derivation in the target collator**

In `ap2_fused_collate_update`, concatenate already-offset short- and long-range tensors in the same order used by the no-target and AP3 collators:

```python
e_ABsr_source_cat = torch.cat(local_e_ABsr_source, dim=0)
e_ABsr_target_cat = torch.cat(local_e_ABsr_target, dim=0)
e_ABlr_source_cat = torch.cat(local_e_ABlr_source, dim=0)
e_ABlr_target_cat = torch.cat(local_e_ABlr_target, dim=0)

e_ABfull_source = torch.cat(
    (e_ABsr_source_cat, e_ABlr_source_cat), dim=0
)
e_ABfull_target = torch.cat(
    (e_ABsr_target_cat, e_ABlr_target_cat), dim=0
)

dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
dimer_ind_lr_cat = torch.cat(
    [data.dimer_ind_lr for data in batch], dim=0
)
dimer_ind_full = torch.cat(
    (dimer_ind_cat, dimer_ind_lr_cat), dim=0
)
```

Store all three full-domain fields in the returned `Data`. Update the collator docstring. Do not add these fields to persisted per-dimer records.

- [ ] **Step 4: Run focused and existing collator tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py::test_target_collate_emits_full_edge_domain -v
python -m pytest tests/test_pt_dataset.py -k "collate" -v
```

Expected: the new contract test passes; existing collator tests remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rackers_thole_damping.py src/apnet_pt/pt_datasets/ap2_fused_ds.py
git commit -m "fix(dataset): emit full dimer edge indices for targets"
```

---

### Task 2: Add and test the geometric combination rule

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `tests/test_polarization.py`
- Modify: `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`

- [ ] **Step 1: Write failing helper tests**

```python
def test_geometric_mean_edge_values_contract():
    source = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    target = torch.tensor([16.0, 25.0], dtype=torch.float64)
    e_source = torch.tensor([0, 1, 2], dtype=torch.long)
    e_target = torch.tensor([1, 0, 1], dtype=torch.long)

    actual = geometric_mean_edge_values(
        source, target, e_source, e_target
    )
    expected = torch.tensor([5.0, 8.0, 15.0], dtype=torch.float64)

    assert torch.equal(actual, expected)
    assert actual.dtype == source.dtype
    assert actual.device == source.device

    exchanged = geometric_mean_edge_values(
        target, source, e_target, e_source
    )
    assert torch.equal(exchanged, expected)


@pytest.mark.parametrize(
    "source,target",
    [
        (
            torch.tensor([1.0, float("nan")]),
            torch.tensor([4.0, 9.0]),
        ),
        (
            torch.tensor([1.0, 4.0]),
            torch.tensor([float("inf"), 9.0]),
        ),
    ],
)
def test_geometric_mean_edge_values_rejects_non_finite(
    source, target
):
    edge = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="finite"):
        geometric_mean_edge_values(source, target, edge, edge)
```

Add a named regression to `tests/test_polarization.py`:

```python
def test_thole_direct_and_mutual_torch_are_finite_and_distinct():
    r = torch.tensor([1.2, 2.5], dtype=torch.float64)
    alpha_i = torch.tensor([0.8, 1.1], dtype=torch.float64)
    alpha_j = torch.tensor([1.3, 0.7], dtype=torch.float64)

    _, direct_l3, direct_l5 = (
        apnet_pt.multipole.thole_damping_direct_torch(
            r, alpha_i, alpha_j, 0.34
        )
    )
    _, mutual_l3, mutual_l5 = (
        apnet_pt.multipole.thole_damping_mutual_torch(
            r, alpha_i, alpha_j, 0.39
        )
    )

    assert torch.isfinite(direct_l3).all()
    assert torch.isfinite(direct_l5).all()
    assert torch.isfinite(mutual_l3).all()
    assert torch.isfinite(mutual_l5).all()
    assert not torch.allclose(direct_l3, mutual_l3)
    assert not torch.allclose(direct_l5, mutual_l5)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "geometric_mean" -v
```

Expected: import or collection failure because `geometric_mean_edge_values` does not exist.

- [ ] **Step 3: Implement the helper**

```python
def geometric_mean_edge_values(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    e_source: torch.Tensor,
    e_target: torch.Tensor,
) -> torch.Tensor:
    if not torch.isfinite(source_values).all():
        raise ValueError("source per-atom values must be finite")
    if not torch.isfinite(target_values).all():
        raise ValueError("target per-atom values must be finite")

    source_edge_values = source_values.index_select(0, e_source)
    target_edge_values = target_values.index_select(0, e_target)
    return torch.sqrt(source_edge_values * target_edge_values)
```

Do not apply `abs`, `clamp`, or device/dtype conversion.

- [ ] **Step 4: Run helper and polarization tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "geometric_mean" -v
python -m pytest tests/test_polarization.py -k "thole" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/apnet_pt/AtomPairwiseModels/mtp_mtp.py tests/test_rackers_thole_damping.py tests/test_polarization.py
git commit -m "feat(polarization): add Rackers geometric combination rule"
```

---

### Task 3: Implement the fixed four-head positive atom model

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`

- [ ] **Step 1: Add deterministic fixtures and failing head tests**

Use a real, tiny nested stack so type validation and tuple preservation are exercised:

```python
@pytest.fixture
def atomic_batch() -> Data:
    return Data(
        x=torch.tensor([8, 1, 1], dtype=torch.long),
        R=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [-0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        molecule_ind=torch.zeros(3, dtype=torch.long),
        total_charge=torch.tensor([0.0], dtype=torch.float32),
        natom_per_mol=torch.tensor([3], dtype=torch.long),
    )


@pytest.fixture
def nested_hfvr_vw_model() -> AtomTypeParamNN:
    atom_model = AtomMPNN(
        n_message=1,
        n_rbf=2,
        n_neuron=8,
        n_embed=4,
        r_cut=5.0,
    )
    nested = AtomTypeParamNN(
        atom_model=atom_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1.0, 0.4],
        param_start_std=[0.0, 0.0],
        n_params=2,
        freeze_atom_model=False,
    )
    set_weights_to_value(nested, 0.01)
    return nested


def test_rackers_parameter_head_contract(
    atomic_batch, nested_hfvr_vw_model
):
    model = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        freeze_atom_model=True,
    )
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.zero_()

    nested_output = nested_hfvr_vw_model(atomic_batch)
    output = model(atomic_batch)
    parameters = output[-1]

    assert parameters.shape == (3, 4)
    assert torch.isfinite(parameters).all()
    assert torch.all(parameters > 0)
    assert torch.allclose(
        parameters.mean(dim=0),
        torch.tensor([1.8, 0.34, 0.39, 1.8]),
        atol=0.05,
    )
    for wrapped, expected in zip(output[:-1], nested_output):
        assert torch.allclose(wrapped, expected)

    parameters.sum().backward()
    for head in model.param_readout_layers:
        gradients = [
            parameter.grad
            for readout in head
            for parameter in readout.parameters()
        ]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    assert all(
        not parameter.requires_grad
        for parameter in model.atom_model.parameters()
    )
    unfrozen = RackersTholeDampingNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        freeze_atom_model=False,
    )
    assert all(
        parameter.requires_grad
        for parameter in unfrozen.atom_model.parameters()
    )


def test_rackers_parameter_head_freeze_and_validation(
    nested_hfvr_vw_model
):
    frozen = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        freeze_atom_model=True,
    )
    assert all(
        not parameter.requires_grad
        for parameter in frozen.atom_model.parameters()
    )

    with pytest.raises(ValueError, match="exactly four"):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_mean=[1.8, 0.34, 0.39],
        )
    with pytest.raises(ValueError, match="exactly four"):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_std=[0.01],
        )
    with pytest.raises(ValueError, match="AtomTypeParamNN"):
        RackersTholeDampingNN(
            atom_model=AtomMPNN(
                n_message=1,
                n_rbf=2,
                n_neuron=8,
                n_embed=4,
            )
        )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "parameter_head" -v
```

Expected: import failure because `RackersTholeDampingNN` and its constants do not exist.

- [ ] **Step 3: Implement the model and configuration**

Add `torch.nn.functional as F`. Validate exact list lengths and nested type before calling `AtomTypeParamNN.__init__`.

Compute raw initialization values with:

```python
def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(tensor)).item()
```

For each requested positive start value `value`, pass
`_inverse_softplus(value - positivity_epsilon)` as the corresponding base-class guess mean. Pass four raw standard deviations and fixed `n_params=4` internally.

After base construction, explicitly enforce the complete nested freeze state:

```python
self.atom_model.requires_grad_(not freeze_atom_model)
```

This must unfreeze parameters previously frozen by an inner wrapper when
`freeze_atom_model=False`; doing nothing in the false branch is incorrect.

Implement forward without changing base rank behavior:

```python
def forward(self, batch):
    output = super().forward(batch)
    raw_parameters = output[-1]
    parameters = F.softplus(raw_parameters) + self.positivity_epsilon
    return (*output[:-1], parameters)
```

Implement `get_config()` with these exact semantic fields:

```python
return {
    "model_type": "RackersTholeDampingNN",
    "parameter_names": list(RACKERS_PARAMETER_NAMES),
    "param_start_mean": list(self.param_start_mean),
    "param_start_std": list(self.param_start_std),
    "positivity_epsilon": self.positivity_epsilon,
    "n_message": self.n_message,
    "n_neuron": self.n_neuron,
    "n_embed": self.n_embed,
    "nested_atom_model": _serialize_nested_atom_model(
        self.atom_model
    ),
}
```

Keep desired positive means in `self.param_start_mean`; store raw means separately so checkpoint metadata never confuses raw and public values.

- [ ] **Step 4: Run head and existing rank/freeze tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "parameter_head" -v
python -m pytest tests/test_freeze_unfreeze.py -k "AtomTypeParamNN" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/apnet_pt/AtomPairwiseModels/mtp_mtp.py tests/test_rackers_thole_damping.py
git commit -m "feat(model): add positive four-head Rackers readout"
```

---

### Task 4: Implement pure and overlap Rackers induction physics

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`

- [ ] **Step 1: Write failing physics-routing tests**

Monkeypatch `thole_damping_direct_torch` and `thole_damping_mutual_torch` to record their per-edge `a` tensors. Use two atoms per monomer and non-identical values so AB, AA, and BB expected values are distinguishable.

The test must assert:

```python
expected_direct = [
    geometric_mean_edge_values(
        direct_A, direct_B, e_AB_source, e_AB_target
    ),
    geometric_mean_edge_values(
        direct_A, direct_A, e_AA_source, e_AA_target
    ),
    geometric_mean_edge_values(
        direct_B, direct_B, e_BB_source, e_BB_target
    ),
]
expected_mutual = [
    geometric_mean_edge_values(
        mutual_A, mutual_B, e_AB_source, e_AB_target
    ),
    geometric_mean_edge_values(
        mutual_A, mutual_A, e_AA_source, e_AA_target
    ),
    geometric_mean_edge_values(
        mutual_B, mutual_B, e_BB_source, e_BB_target
    ),
]
```

Run the pure kernel twice with different `ind_overlap_A/B` and assert identical energies. Run the overlap kernel and calculate the expected difference using the existing formula:

```python
dR_AB, _ = get_distances(
    RA, RB, e_AB_source, e_AB_target
)
dR_AB = dR_AB / constants.au2ang
sigma_A = valence_widths_A.index_select(0, e_AB_source)
sigma_B = valence_widths_B.index_select(0, e_AB_target)
B_ij = torch.sqrt(1.0 / (sigma_A * sigma_B))
S_ij = (
    (B_ij * dR_AB) ** 2 / 3.0
    + B_ij * dR_AB
    + 1.0
) * torch.exp(-B_ij * dR_AB)
expected_overlap = (
    ind_overlap_A.index_select(0, e_AB_source)
    * S_ij
    * ind_overlap_B.index_select(0, e_AB_target)
    * constants.h2kcalmol
)
assert torch.allclose(
    pure_energy - overlap_energy,
    expected_overlap,
    atol=1e-6,
)
```

Also assert direct and mutual monkeypatch call lists each contain exactly three
calls ordered AB, AA, BB. Add an effect-level sentinel test around the tensor
consumption seam: changing only direct AA values must change A's permanent-field
contribution, changing only direct BB values must change B's permanent-field
contribution, and changing only mutual values must leave the initial permanent
field unchanged while changing an SCF update. The test must fail if direct AA/BB
tensors are merely constructed and discarded.

- [ ] **Step 2: Run the routing test and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "kernel_routes" -v
```

Expected: failure because `rackers_thole_induction` does not exist.

- [ ] **Step 3: Implement the shared Rackers kernel**

Add a private tensor builder which accepts already-combined edge values:

```python
def _rackers_distance_tensors(
    Ri: torch.Tensor,
    Rj: torch.Tensor,
    e_source: torch.Tensor,
    e_target: torch.Tensor,
    alpha_i: torch.Tensor,
    alpha_j: torch.Tensor,
    thole_edge_values: torch.Tensor,
    damping_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```

It must:

1. Use `get_distances`.
2. Convert distances and displacement vectors with `constants.au2ang`.
3. Select edge polarizabilities.
4. Call only `thole_damping_direct_torch` for `"direct"` or only `thole_damping_mutual_torch` for `"mutual"`.
5. Build `T1` and `T2` with the same units and einsums as the existing optimized kernel.
6. Raise `ValueError` for any other damping type.

In `rackers_thole_induction`:

- Compute Hirshfeld-scaled polarizabilities exactly as in `induced_dipole_induction_optimized`.
- Compute direct and mutual geometric means separately for AB, AA, and BB.
- Build direct AB/AA/BB tensors and consume all three sets in permanent
  charge/dipole fields contributing to `mu_induced_0_A/B`; direct AA contributes
  only to A's initial field and direct BB contributes only to B's initial field.
- Build mutual AB/AA/BB tensors and use only their `T2` tensors during SCF.
- Keep the initial permanent-field construction in a narrow helper or otherwise
  expose a test seam so effect-level AA/BB direct routing is verifiable rather
  than inferred from damping-helper call counts.
- Use the direct AB tensors for final permanent-induced `E_qu` and `E_uu`.
- Retain existing iteration count, convergence threshold, mixing, units, and scatter dimensions.
- If `include_overlap` is false, never index or multiply `ind_overlap_A/B`.
- If `include_overlap` is true, reproduce the existing `B_ij`, `S_ij`, and `K_A * S_ij * K_B * h2kcalmol` calculation, then subtract it once.
- Do not modify `induced_dipole_induction`, `induced_dipole_induction_optimized`, or `induced_dipole_induction_optimized_no_correction`.

Do not decorate the new kernel with `torch.compile` until focused monkeypatch and gradient tests pass.

- [ ] **Step 4: Run routing and existing Thole tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "kernel_routes" -v
python -m pytest tests/test_polarization.py -k "thole" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/apnet_pt/AtomPairwiseModels/mtp_mtp.py tests/test_rackers_thole_damping.py
git commit -m "feat(physics): add Rackers direct and mutual induction kernel"
```

---

### Task 5: Add both `DimerProp` modes and joint gradient behavior

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`

- [ ] **Step 1: Write failing mode, mutation, and gradient tests**

Add a controlled routing test parameterized over `elst_damping_type="CLIFF"`
and `"AMOEBA"`. Monkeypatch the matching `mtp_elst_damping` or
`mtp_elst_damping_AMOEBA` function to mutate its charge argument and record its K
columns and edges. Monkeypatch `rackers_thole_induction` to record induction
arguments.

Assert for both modes and both electrostatic damping types:

- Exactly the selected CLIFF or AMOEBA electrostatic function is called; the
  other is not called.
- Electrostatics receives columns 0 as per-atom values.
- Induction receives columns 1, 2, and 3 separately.
- Induction receives
  `abs(output_A[-2][:, 0])`/`abs(output_B[-2][:, 0])` as HFVR and
  `output_A[-2][:, 1]`/`output_B[-2][:, 1]` as valence widths, without swapping
  the columns.
- Both functions receive `e_ABfull_source/target`.
- The pure mode passes `include_overlap=False`.
- The overlap mode passes `include_overlap=True`.
- Charge mutation inside the electrostatic stub does not alter the charge tensor received by the induction stub.
- The returned edge tensor has shape `[len(dimer_ind_full), 2]`.

Add a sentinel test that changes only valence widths while holding all other
inputs fixed: `rackers_thole` energy must remain unchanged, while
`rackers_thole_overlap` energy must change. This proves VW is carried for base
compatibility but is energy-active only in the overlap variant.

Add a real CPU test using the tiny nested model and a two-dimer batch:

```python
@pytest.mark.parametrize(
    "mode,expected_active_heads",
    [
        ("rackers_thole", {0, 1, 2}),
        ("rackers_thole_overlap", {0, 1, 2, 3}),
    ],
)
def test_rackers_joint_forward_scatter_and_gradients(
    mode,
    expected_active_heads,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
):
    model = RackersTholeDampingNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        n_message=1,
        n_neuron=8,
        n_embed=4,
        freeze_atom_model=True,
    )
    dimer = DimerProp(
        ATParam=model,
        dimer_eval=mode,
        freeze_atom_model=True,
    )

    edge_energy, output_A, output_B = dimer(
        synthetic_dimer_batch
    )
    assert edge_energy.shape == (
        synthetic_dimer_batch.e_ABfull_source.numel(),
        2,
    )
    assert torch.isfinite(edge_energy).all()

    dimer_energy = scatter_sum_compile(
        edge_energy,
        synthetic_dimer_batch.dimer_ind_full,
        dim_size=synthetic_dimer_batch.total_charge_A.size(0),
    )
    assert dimer_energy.shape == (2, 2)
    assert torch.isfinite(dimer_energy).all()

    dimer_energy.square().mean().backward()
    for index, head in enumerate(model.param_readout_layers):
        gradients = [
            parameter.grad
            for readout in head
            for parameter in readout.parameters()
        ]
        has_nonzero_gradient = any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient) > 0
            for gradient in gradients
        )
        assert has_nonzero_gradient == (
            index in expected_active_heads
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    updated_parameters = model(
        synthetic_dimer_batch.batch_atomic_A
    )[-1]
    assert torch.isfinite(updated_parameters).all()
    assert torch.all(updated_parameters > 0)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "dimer_forward or joint_forward" -v
```

Expected: `DimerProp.set_forward` rejects both Rackers modes.

- [ ] **Step 3: Implement shared `DimerProp` forwarding**

Extend `set_forward`:

```python
elif dimer_eval == "rackers_thole":
    self.forward = self._rackers_thole_forward
    self.polarizability_table = constants.polarizability_table.clone()
elif dimer_eval == "rackers_thole_overlap":
    self.forward = self._rackers_thole_overlap_forward
    self.polarizability_table = constants.polarizability_table.clone()
```

Add thin wrappers calling one private common method:

```python
def _rackers_thole_forward(self, batch):
    return self._rackers_thole_common_forward(
        batch, include_overlap=False
    )


def _rackers_thole_overlap_forward(self, batch):
    return self._rackers_thole_common_forward(
        batch, include_overlap=True
    )
```

The common method must:

- Evaluate `self.AtomTypeParam` once per monomer.
- Bind `parameters_A = output_A[-1]` and `parameters_B = output_B[-1]`.
- Bind `hfvr_A/B = abs(output_A/B[-2][:, 0])` and
  `valence_widths_A/B = output_A/B[-2][:, 1]`, then pass those exact tensors to
  the induction kernel.
- Use the named index constants.
- Select `mtp_elst_damping_AMOEBA` only when
  `self.elst_damping_type == "AMOEBA"`, select `mtp_elst_damping` for `"CLIFF"`,
  and raise `ValueError` for unsupported values so checkpoint metadata cannot
  disagree with runtime physics.
- Pass `output_A[0].clone()` and `output_B[0].clone()` to electrostatics.
- Pass the untouched original charge tensors to induction.
- Use `e_ABfull_source/target`.
- Return `torch.vstack((Elst, Indu)).T, output_A, output_B`.
- Never use negative-index column arithmetic beyond retrieving the final parameter matrix.
- Never apply `abs` or clamping to Rackers columns.

- [ ] **Step 4: Run focused mode tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "dimer_forward or joint_forward" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/apnet_pt/AtomPairwiseModels/mtp_mtp.py tests/test_rackers_thole_damping.py
git commit -m "feat(model): add Rackers dimer evaluation modes"
```

---

### Task 6: Add shared harnesses, full-edge training, and standalone checkpoints

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py`

- [ ] **Step 1: Write failing harness and checkpoint tests**

Add parameterized constructor tests for both harness classes. Verify fixed model type, fixed mode, four outputs, default freeze, and rejection of incompatible initialization lengths.

Add checkpoint round-trip tests:

```python
@pytest.mark.parametrize(
    "harness_type,expected_mode",
    [
        (RackersTholeDampingModel, "rackers_thole"),
        (
            RackersTholeDampingOverlapModel,
            "rackers_thole_overlap",
        ),
    ],
)
def test_rackers_checkpoint_round_trip(
    tmp_path,
    harness_type,
    expected_mode,
    nested_hfvr_vw_model,
    synthetic_qcel_dimers,
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    before = harness.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )

    checkpoint_path = tmp_path / f"{expected_mode}.pt"
    harness.save_model(checkpoint_path)

    checkpoint = model_io.load_checkpoint(checkpoint_path)
    assert checkpoint["model_type"] == "RackersTholeDampingNN"
    assert checkpoint["config"]["parameter_names"] == list(
        RACKERS_PARAMETER_NAMES
    )
    assert checkpoint["config"]["dimer_eval"] == expected_mode

    loaded = harness_type(
        pre_trained_model_path=checkpoint_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    after = loaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(before, after, atol=1e-6)

    second_path = tmp_path / f"{expected_mode}-second.pt"
    loaded.save_model(second_path)
    reloaded = harness_type(
        pre_trained_model_path=second_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert reloaded.model.get_config() == loaded.model.get_config()
    second_predictions = reloaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(after, second_predictions, atol=1e-6)
```

Tamper a copied checkpoint and assert failures for:

- Reordered `parameter_names`.
- Missing `parameter_names`.
- `model_type="AtomTypeParamNN"`.
- Loading a pure checkpoint through the overlap harness or vice versa.

Add a training aggregation test that calls the private batch loop through a one-epoch CPU harness or a narrow test seam and makes `dimer_ind` intentionally incompatible with the full-edge output while `dimer_ind_full` matches. This test must fail unless training and evaluation select `dimer_ind_full` for both Rackers modes.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "harness or checkpoint or training_uses_full" -v
```

Expected: missing harness classes and/or failure to reconstruct Rackers checkpoints.

- [ ] **Step 3: Implement recursive Rackers checkpoint metadata**

Add Rackers-only recursive helpers:

```python
def _serialize_nested_atom_model(model: nn.Module) -> dict:
```

Support `AtomMPNN` and recursive `AtomTypeParamNN`. Each node records `model_type`, its `get_config()`, and nested `atom_model` metadata where present.

```python
def _rebuild_nested_atom_model(
    metadata: dict,
    freeze_atom_model: bool,
) -> nn.Module:
```

Reconstruct only supported types and raise a clear `ValueError` otherwise. State restoration still comes from the top-level Rackers state dict.

Extend the `AM_DimerParam_Model` model-construction branches to recognize `model_type == "RackersTholeDampingNN"` without changing the existing `AtomTypeParamNN` branches. On checkpoint load:

1. Call `model_io.validate_checkpoint` with expected type `RackersTholeDampingNN`.
2. Validate exact parameter names before constructing any model.
3. Validate `dimer_eval` against the requested harness.
4. Rebuild the nested model from `nested_atom_model`.
5. Construct `RackersTholeDampingNN` from checkpoint config.
6. Load the complete state dict strictly.
7. Treat checkpoint architecture and initialization metadata as authoritative:
   skip the inherited post-construction override block for `n_message`,
   `n_neuron`, `n_embed`, means, and standard deviations whenever
   `pre_trained_model_path` is supplied. Caller defaults must not mutate loaded
   architecture attributes without rebuilding their layers.

Ensure `_create_checkpoint` writes:

```python
model_config["dimer_eval"] = self.dimer_eval_type
model_config["elst_damping_type"] = self.elst_damping_type
```

Keep existing non-Rackers keys and loading paths unchanged.

- [ ] **Step 4: Implement shared harnesses and full-edge training**

Add a private base after `AM_DimerParam_Model`:

```python
class _RackersTholeDampingModelBase(AM_DimerParam_Model):
    DIMER_EVAL: str

    def __init__(
        self,
        dataset=None,
        atom_model: AtomTypeParamNN | None = None,
        pre_trained_model_path=None,
        n_message: int = 3,
        n_neuron: int = 64,
        n_embed: int = 8,
        param_start_mean=RACKERS_INITIAL_VALUES,
        param_start_std=RACKERS_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        freeze_atom_model: bool = True,
        **dataset_kwargs,
    ):
```

The private base must fix `atom_model_type`, `model_type`, `n_params=4`, and `dimer_eval_type`, validate four-value settings, and delegate to `AM_DimerParam_Model`. The two public subclasses only set `DIMER_EVAL`.

Add a private aggregation selector to `AM_DimerParam_Model`:

```python
def _dimer_index_for_output(self, batch):
    if self.dimer_eval_type in {
        "rackers_thole",
        "rackers_thole_overlap",
    }:
        return batch.dimer_ind_full
    return batch.dimer_ind
```

Use it in training, evaluation, and `predict_qcel_mols_dimer` aggregation. Add
both Rackers modes to the combined `[0, 2]` target branch in
`single_proc_train`. Add a Rackers prediction test with nonempty long-range
edges proving full-edge output and `dimer_ind_full` lengths match, plus a legacy
short-range mode regression proving `_dimer_index_for_output` returns
`dimer_ind` and matches that forward's edge count. Preserve all other existing
mode behavior.

- [ ] **Step 5: Run checkpoint, training, and compatibility tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "harness or checkpoint or training_uses_full" -v
python -m pytest tests/test_model_io.py -v
python -m pytest tests/test_freeze_unfreeze.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/apnet_pt/AtomPairwiseModels/mtp_mtp.py tests/test_rackers_thole_damping.py
git commit -m "feat(model): add Rackers harnesses and checkpoint contract"
```

---

### Task 7: Add both `train_models.py` dispatch routes and CLI defaults

**Files:**
- Modify: `tests/test_rackers_thole_damping.py`
- Modify: `train_models.py`

- [ ] **Step 1: Write failing dispatch/default tests**

Monkeypatch:

- `AtomPairwiseModels.mtp_mtp.AtomTypeParamModel`
- `RackersTholeDampingModel`
- `RackersTholeDampingOverlapModel`

Use fakes that record constructor arguments and expose a named `train` signature matching supported harness arguments.

Parameterize over both identifiers and assert:

- The correct harness is selected.
- `am_model_path` is forwarded as `atom_model_pre_trained_path` when constructing HFVR/VW.
- `atom_type_param_model_path` is forwarded as its pretrained path.
- The low-level HFVR/VW model is passed to the Rackers harness.
- Default means/stds are the four Rackers values.
- `freeze_atom_model=True` by default and false when requested, both when
  constructing the HFVR/VW wrapper and the outer Rackers harness.
- Real nested parameters are all frozen by default and all trainable on the
  unfreeze route; checking only recorded Boolean kwargs is insufficient.
- The selected harness receives its training call.
- `n_params` does not reach the Rackers harness.

Add validation tests:

```python
@pytest.mark.parametrize(
    "field,value",
    [
        ("param_start_mean", 1.8),
        ("param_start_std", 0.01),
        ("param_start_mean", [1.8, 0.34, 0.39]),
        ("param_start_std", [0.01, 0.01, 0.01]),
    ],
)
def test_rackers_dispatch_rejects_ambiguous_parameter_lists(
    field, value
):
    kwargs = {
        "apnet_model_type": "RackersTholeDampingModel",
        "pre_trained_model_path": None,
        "param_start_mean": [1.8, 0.34, 0.39, 1.8],
        "param_start_std": [0.01, 0.01, 0.01, 0.01],
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="exactly four"):
        train_models.train_pairwise_model(**kwargs)
```

Add direct tests showing existing routes still broadcast scalar values.

- [ ] **Step 2: Run dispatch tests and verify RED**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "dispatch" -v
```

Expected: invalid model type or missing Rackers class selection.

- [ ] **Step 3: Implement default resolution and dispatch**

Define in `train_models.py`:

```python
RACKERS_MODEL_TYPES = {
    "RackersTholeDampingModel",
    "RackersTholeDampingOverlapModel",
}
RACKERS_PARAM_START_MEAN = [1.8, 0.34, 0.39, 1.8]
RACKERS_PARAM_START_STD = [0.01, 0.01, 0.01, 0.01]
```

Change `train_pairwise_model` defaults to `param_start_mean=None` and `param_start_std=None`. Resolve them before generic scalar broadcasting:

- Rackers + `None`: copy the Rackers list.
- Rackers + scalar or non-four list: raise `ValueError`.
- Non-Rackers + `None`: restore direct-call defaults `1.5` and `0.1`.
- Non-Rackers + scalar: retain existing broadcasting by `n_params`.

Select exact harness classes in the model dispatch. Add a dedicated Rackers construction branch that first constructs:

```python
atom_type_hf_vw_model = (
    AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_model_path,
        pre_trained_model_path=atom_type_param_model_path,
        freeze_atom_model=freeze_atom_model,
    )
)
```

Then construct the selected harness with its `.model`, dataset and architecture
settings, `pretrained_model`, four initialization values, damping type, random
seed, and the same `freeze_atom_model` value. The Rackers constructor must
explicitly apply `requires_grad_(not freeze_atom_model)` to its complete nested
model so an unfreeze request reverses any prior frozen flags.

Change parser defaults for both parameter options to `None`. Parse only non-`None` strings. Before calling `train_pairwise_model`, preserve historical CLI defaults for non-Rackers routes by resolving unset values to `2.0` and `0.1`; leave Rackers unset so the function chooses Rackers defaults.

Update `--train_apnet`, parameter option, and `--unfreeze_atom_model` help text to name both Rackers identifiers and route-specific behavior.

- [ ] **Step 4: Run dispatch and help tests and verify GREEN**

```bash
python -m pytest tests/test_rackers_thole_damping.py -k "dispatch" -v
python train_models.py --help
```

Confirm help output contains both exact identifiers.

- [ ] **Step 5: Commit**

```bash
git add train_models.py tests/test_rackers_thole_damping.py
git commit -m "feat(training): dispatch Rackers damping variants"
```

## Final Verification

Run the full focused suite and required regressions:

```bash
python -m pytest tests/test_rackers_thole_damping.py -v
python -m pytest tests/test_atomtype_props.py::test_elst_multipoles_MTP_torch_damping_AM_DimerParam -v
python -m pytest tests/test_polarization.py -k "thole" -v
python -m pytest tests/test_model_io.py -v
python -m pytest tests/test_pt_dataset.py -k "collate" -v
python -m pytest tests/test_freeze_unfreeze.py -v
python train_models.py --help
git status --short
```

Acceptance requires all pytest commands to pass, both identifiers to appear in help output, and `git status --short` to be empty after the seven task commits.

## Files to Modify

- `src/apnet_pt/pt_datasets/ap2_fused_ds.py` - full-edge target collation.
- `src/apnet_pt/AtomPairwiseModels/mtp_mtp.py` - Rackers model, physics, modes, harnesses, training aggregation, and checkpoints.
- `train_models.py` - both routes, defaults, validation, and CLI help.
- `tests/test_polarization.py` - named direct/mutual Thole regression.

## New Files

- `tests/test_rackers_thole_damping.py` - focused CPU-only feature coverage.

## Dependencies

1. Task 2 depends only on existing multipole damping helpers.
2. Task 3 depends on Task 2 constants only conceptually; it can use the helper independently.
3. Task 4 depends on Tasks 2 and 3 for semantic constants and combination behavior.
4. Task 5 depends on Tasks 3 and 4.
5. Task 6 depends on Tasks 1, 3, and 5 for full-edge batches, model construction, and modes.
6. Task 7 depends on Task 6 public harness classes.
7. Final verification depends on all seven committed tasks.

## Risks

- `torch.isfinite(...).all()` introduces Python-side validation; keep the combination helper outside compiled SCF internals if compilation becomes graph-breaking.
- The pure variant intentionally gives column 3 no energy gradient. Tests must expect only heads 0–2 to be active there and all four in the overlap variant.
- Target and no-target collators must preserve identical full-edge ordering: all short-range edges followed by all long-range edges.
- Standalone reconstruction must serialize the complete nested `AtomTypeParamNN → AtomMPNN` architecture, not merely the outer HFVR/VW configuration.
- Rackers checkpoint validation must execute before state loading so reordered physical meanings cannot be hidden by shape-compatible tensors.
- Existing models must continue using `dimer_ind`; only Rackers modes may switch training aggregation to `dimer_ind_full`.
- No blocking design ambiguity remains. The updated approved spec explicitly defines four heads and separates pure-IPD from overlap-enabled behavior.
