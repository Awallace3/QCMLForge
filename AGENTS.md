# AGENTS.md — QCMLForge Agent Guide

QCMLForge (`qcmlforge`) is a PyTorch package for quantum-chemistry ML, primarily SAPT interaction-energy prediction. Its main package, `apnet_pt`, contains atomic multipole models and AP-Net pairwise models.

## Anti-Patterns

Keep this section near the top and extend it as new recurring problems are identified. Add each rule as a concise, actionable bullet explaining what to avoid; include the preferred alternative when useful.

- Do not use monkeypatch logic in production code or tests.
- Do not loop over atoms when a vectorized NumPy/PyTorch operation is practical.
- Avoid introducing graph-breaking operations into code intended for `torch.compile()`.
- Do not commit exploratory scripts; put throwaway work in the ignored `agent_scratch/` directory.

## Architecture

Every model has a low-level `nn.Module` and a high-level training/inference harness:

| Module | Harness | Purpose |
|---|---|---|
| `AtomMPNN` | `AtomModel` | Predict atomic charges, dipoles, and quadrupoles |
| `APNet2` | `APNet2Model` | Predict dimer SAPT components |
| `APNet3` | `APNet3_AtomType_Model` | AP-Net with atom-type parameters |
| `dAPNet2_MPNN` | `dAPNet2Model` | Predict a SAPT delta correction on top of APNet2 |

Main layout:

```text
src/apnet_pt/
├── constants.py
├── util.py
├── layers.py
├── multipole.py
├── AtomModels/
├── AtomPairwiseModels/
└── pt_datasets/
```

## Setup and Commands

```bash
conda env create -f environment.yml
conda activate qcml
pip install -e .

python -m pytest tests/
python -m pytest tests/test_ap2.py
python -m pytest tests/test_ap2.py::test_ap2_architecture -v
python -m pytest tests/ -k "am" -v
python -m pytest tests/ --cov=src/apnet_pt
```

Training examples:

```bash
python train_models.py --train_am AtomModel --am_model_path ./models/am_example.pt --n_epochs_atom 500
python train_models.py --train_apnet APNet2 --ap_model_path ./models/ap2_example.pt --n_epochs 50
```

## Code Standards

- Follow PEP 8: four-space indentation and approximately 88-character lines.
- Use `snake_case` for functions and modules, `PascalCase` for classes, and `ALL_CAPS` for constants.
- Group imports as standard library, third-party, then local/relative imports, separated by blank lines.
- Prefer relative imports within the package.
- Add type hints to public APIs.
- Use NumPy-style docstrings for public or non-obvious interfaces.
- Keep functions focused and preferably under 50 lines; extract helpers from complex logic.
- Implement PyTorch models with the standard `nn.Module`/`forward()` pattern.

Error handling:

- Return `None` when invalid input is an expected outcome, such as a non-dimer passed to `qcel_to_dimerdata`.
- Use descriptive assertions only for internal invariants.
- Raise specific exceptions such as `ValueError` for invalid user input.

Performance:

- Prefer vectorized NumPy/PyTorch operations.
- Support GPU execution where appropriate.
- Keep production paths compatible with `torch.compile()`.
- Prefer `scatter_sum_compile()` over `torch_geometric.utils.scatter` in compiled code.

## Development Workflow

Use test-driven development for new behavior:

1. Add or update a focused test in `tests/test_*.py`.
2. Implement the smallest correct change.
3. Run the focused test, then the relevant broader suite.

Tests should use deterministic inputs and weights where practical. Use `pytest.mark.skip` only with a clear reason. Put experiments, timing scripts, and disposable diagnostics in `agent_scratch/`.

## Common APIs

Create molecules with `qcelemental`:

```python
import qcelemental as qcel

mol = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
--
0 1
O 3.000000 0.500000 0.000000
units angstrom
""")
```

Load pretrained atomic models and predict pair energies through the harnesses:

```python
import torch
from apnet_pt import APNet2Model
from apnet_pt.AtomModels.ap2_atom_model import AtomModel

atom_model = AtomModel(
    ds_root=None,
    ignore_database_null=True,
    use_GPU=torch.cuda.is_available(),
).set_pretrained_model(model_id=0)

pair_model = APNet2Model(
    atom_model=atom_model.model,
    ignore_database_null=True,
    use_GPU=torch.cuda.is_available(),
)
energies = pair_model.predict_qcel_mols([mol], batch_size=1)
# Each result is [elst, exch, indu, disp] in kcal/mol.
```

## Key Dependencies

- PyTorch: model implementation and training
- PyTorch Geometric: graph operations
- QCEngine/QCElemental: molecular data structures
- QCPortal: QCArchive access
- NumPy: numerical operations
