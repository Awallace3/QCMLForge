User Guide
===============

QCMLForge has two main surfaces:

- `qcmlforge` for QCArchive and QCFractal setup helpers.
- `apnet_pt` for AP-Net datasets, pretrained inference helpers, and training
  harnesses.

Core Workflows
--------------

Pretrained inference
~~~~~~~~~~~~~~~~~~~~

Use the helpers in `apnet_pt.pretrained_models` when you want the shipped
ensembles without manually wiring model classes:

- `atom_model_predict` predicts atomic charges, dipoles, and quadrupoles.
- `apnet2_model_predict` predicts total and component interaction energies.
- `apnet2_model_predict_pairs` returns pairwise and fragment-pair breakdowns.

Manual model access
~~~~~~~~~~~~~~~~~~~

For lower-level control, instantiate the model harnesses directly. Common entry
points include:

- `apnet_pt.AtomModels.ap2_atom_model.AtomModel`
- `apnet_pt.APNet2Model`
- `apnet_pt.AtomPairwiseModels.apnet3.APNet3Model`
- `apnet_pt.AtomPairwiseModels.dapnet2.DAPNet2Model`

The harness classes manage dataset loading, pretrained weights, prediction
interfaces, and training loops around the underlying `torch.nn.Module`
implementations.

Data and Molecule Inputs
------------------------

Inference helpers and model wrappers generally accept `qcelemental`
`Molecule` objects. For dimer predictions, provide molecules with exactly two
fragments separated by `--` in the input geometry.

Several utility functions in `apnet_pt.util` and dataset helpers in
`apnet_pt.atomic_datasets` and `apnet_pt.pairwise_datasets` support converting
QCElemental molecules into PyTorch Geometric data objects.

Training Scripts
----------------

`train_models.py` is the main command-line entry point for local training. The
repository README includes minimal examples for:

- AtomModel multipole training with `--train_am <model_name>`
- APNet interaction training with `--train_apnet <model_name>`

Use `train_ddp_slurm.py` when running distributed jobs under Slurm.

Model Guide
-----------

See :doc:`apnet_pt_models` for the trainable `apnet_pt` classes exposed by
`train_models.py`, including their expected inputs, outputs, and submodel
dependencies.

QCArchive Utilities
-------------------

The `qcmlforge.qca.setup_qcarchive_qcfractal` helper bootstraps a local
QCFractal configuration, database layout, and compute-manager resources file.
Use it when you need a lightweight local QCArchive deployment for dataset
generation or experimentation.

Validation
----------

The `tests/` directory contains architecture, dataset, model I/O,
polarization, fused-model, and classical-component coverage. Run the full suite
with:

.. code-block:: bash

   python -m pytest tests/
