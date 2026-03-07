# AGENTS.md - Guidelines for QCMLForge

This document provides comprehensive guidelines for AI coding agents working with the QCMLForge codebase - a PyTorch implementation of quantum chemistry machine learning models including AP-Net variants for molecular interaction energy prediction.

## Project Overview

QCMLForge (`qcmlforge`) builds ML models for quantum chemistry, specifically for predicting SAPT (Symmetry-Adapted Perturbation Theory) interaction energies. The core package is `apnet_pt` which contains PyTorch implementations of:
- **AtomModels**: Single-molecule models predicting atomic multipoles (charges, dipoles, quadrupoles)
- **AtomPairwiseModels**: Dimer interaction models (APNet2, APNet3, dAPNet2) predicting SAPT energy components

## Model Architecture Pattern

For every PyTorch model in this repository, there is a low-level `nn.Module` class and a higher-level harness/wrapper class:

| Low-level Model | High-level Harness | Purpose |
|-----------------|-------------------|---------|
| `AtomMPNN` | `AtomModel` | Predicts atomic multipoles from molecular geometry |
| `APNet2` | `APNet2Model` | Predicts SAPT energies for molecular dimers |
| `APNet3` | `APNet3_AtomType_Model` | Extended APNet with atom-type parameters |

The harness classes handle dataset loading, training loops, and prediction interfaces.

## Build/Lint/Test Commands

### Installation
```bash
# Conda environment (recommended)
conda env create -f environment.yml
conda activate qcml

# Editable install
pip install -e .
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run single test file
python -m pytest tests/test_ap2.py

# Run specific test function with verbose output
python -m pytest tests/test_ap2.py::test_ap2_architecture -v

# Run tests with coverage
python -m pytest tests/ --cov=src/apnet_pt

# Run tests matching a pattern
python -m pytest tests/ -k "am" -v
```

### Training Models
```bash
# Train AtomModel (single-molecule multipole prediction)
python train_models.py --train_am AtomModel --am_model_path ./models/am_example.pt --n_epochs_atom 500

# Train APNet2 (dimer interaction energy)
python train_models.py --train_apnet APNet2 --ap_model_path ./models/ap2_example.pt --n_epochs 50

# Train with custom hyperparameters
python train_models.py --train_am AtomModel --lr 5e-4 --n_rbf_atom 8 --n_neuron_atom 128 --r_cut_atom 5.0
```

## Code Style Guidelines

### Import Order
Organize imports in three groups separated by blank lines:
1. Standard library
2. Third-party packages
3. Local/relative imports

```python
# Standard library
import os
import time
import warnings

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import qcelemental as qcel
from torch_geometric.data import Data

# Local imports (use relative within package)
from .. import multipole
from ..AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt import constants
```

### Naming Conventions
- **Functions/methods**: `snake_case` (e.g., `load_dimer_dataset`, `predict_qcel_mols`)
- **Classes**: `PascalCase` (e.g., `AtomModel`, `APNet2Model`, `DistanceLayer`)
- **Constants**: `ALL_CAPS` in `constants.py` (e.g., `au2ang`, `h2kcalmol`, `elem_to_z`)
- **Module files**: `snake_case` (e.g., `ap2_atom_model.py`, `pairwise_datasets.py`)

### Docstrings (NumPy Style)
```python
def scatter_sum_compile(src, index, dim_size, reduce="sum"):
    """
    Compile-friendly version of torch_geometric scatter for sum reduction.
    
    Parameters
    ----------
    src : torch.Tensor
        Source tensor to scatter (any number of dimensions)
    index : torch.Tensor
        Index tensor indicating where to scatter each element (1D)
    dim_size : int
        Size of the output dimension 0
    reduce : str
        Reduction operation, must be "sum" or "add"
    
    Returns
    -------
    torch.Tensor
        Scattered tensor with shape (dim_size, *src.shape[1:])
    
    Notes
    -----
    Only supports reduce="sum" and reduce="add" which are equivalent.
    """
```

### Type Hints
Add type hints for function signatures, especially for public APIs:
```python
def qcel_to_dimerdata(dimer: qcel.models.Molecule) -> tuple | None:
    ...
```

### Error Handling
- Return `None` for invalid inputs when appropriate (e.g., `qcel_to_dimerdata`)
- Use assertions for internal invariants with descriptive messages
- Raise specific exceptions (`ValueError`, `AssertionError`) with clear messages

```python
if len(dimer.fragments) != 2:
    return None  # Invalid dimer

assert abs(tQ - aQ * n) < 1e-6, "Charge mismatch detected"

if atom_model_type not in valid_types:
    raise ValueError(f"Invalid Atom Model Type: {atom_model_type}")
```

### Code Structure
- **PEP 8 compliance**: 4-space indentation, ~88 char line length
- **Functions**: Keep under 50 lines when possible; extract helpers for complex logic
- **Classes**: Use `nn.Module` pattern for PyTorch models with `forward()` method

### File Organization
```
src/apnet_pt/
    __init__.py
    constants.py         # Physical constants, element mappings
    util.py              # Data loading, molecule conversion utilities
    layers.py            # Reusable neural network layers
    multipole.py         # Multipole math operations
    AtomModels/          # Single-molecule models
    AtomPairwiseModels/  # Dimer interaction models
    pt_datasets/         # Dataset classes
```

### Performance Guidelines
- Use vectorized NumPy/PyTorch operations; avoid Python loops over atoms
- Leverage GPU when available (`use_GPU=True` parameter)
- Use `torch.compile()` for production; avoid graph-breaking operations
- Prefer `scatter_sum_compile()` over `torch_geometric.utils.scatter` for compilation

## Development Workflow

### Test-Driven Development (TDD)
New features should follow TDD:
1. Write pytest test first in `tests/test_*.py`
2. Implement code to pass the test
3. Iterate on both until complete

### Test Patterns
```python
import pytest
import torch
import numpy as np
import qcelemental as qcel

def test_model_architecture():
    """Test model produces expected outputs with known weights."""
    model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    set_weights_to_value(model.model, 0.01)  # Deterministic weights
    output = model.predict(...)
    assert np.allclose(output, expected, atol=1e-6)

@pytest.mark.skip("Reason for skipping")
def test_not_yet_implemented():
    pass
```

### Throwaway Code Directory
Use `agent_scratch/` for quick experiments, timing tests, or exploratory scripts that should NOT be committed to git history:
```bash
# This directory is in .gitignore
python agent_scratch/test_timing.py
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.7.0 | Core deep learning framework |
| `torch-geometric` | 2.6.1 | Graph neural network layers |
| `qcelemental` | >=0.29.0 | QC molecule/data structures |
| `qcportal` | >=0.59 | QCArchive database access |
| `numpy` | >=2.2.0 | Numerical operations |

## Common Patterns

### Creating Test Molecules
```python
mol = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
units angstrom
""")
```

### Loading Pretrained Models
```python
atom_model = AtomModel(
    ds_root=None,
    ignore_database_null=True,
    use_GPU=torch.cuda.is_available(),
).set_pretrained_model(model_id=0)
```

### Making Predictions
```python
pair_model = APNet2Model(
    atom_model=atom_model.model,
    ignore_database_null=True,
    use_GPU=True,
)
energies = pair_model.predict_qcel_mols([mol], batch_size=1)
# Returns: [elst, exch, indu, disp] in kcal/mol
```

