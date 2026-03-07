Getting Started
===============

QCMLForge combines QCArchive utilities in `qcmlforge` with PyTorch AP-Net
models in `apnet_pt`. This guide mirrors the README examples so the quickest
path from install to prediction lives in one place.

Installation
------------

Create the recommended Conda environment and install the package in editable
mode:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate qcml
   pip install -e .

If `torch-scatter` or `torch-geometric` fail to import, reinstall the matching
prebuilt wheels for your PyTorch runtime:

.. code-block:: bash

   # CUDA example
   pip uninstall torch-geometric torch-scatter
   export TORCH=2.7.0
   export CUDA=cu126
   pip install torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

   # CPU example
   pip uninstall torch-geometric torch-scatter
   export TORCH=2.7.0
   pip install torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
   pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html

Run Pretrained AtomModel Inference
----------------------------------

The README multipole example uses `apnet_pt.pretrained_models.atom_model_predict`
to evaluate one or more monomers:

.. code-block:: python

   import apnet_pt
   import qcelemental

   mol_mon = qcelemental.models.Molecule.from_data("""0 1
   16  -0.8795  -2.0832  -0.5531
   7   -0.2959  -1.8177   1.0312
   7    0.5447  -0.7201   1.0401
   6    0.7089  -0.1380  -0.1269
   6    0.0093  -0.7249  -1.1722
   1    1.3541   0.7291  -0.1989
   1   -0.0341  -0.4523  -2.2196
   units angstrom
   """)

   mols = [mol_mon for _ in range(3)]
   multipoles = apnet_pt.pretrained_models.atom_model_predict(
       mols,
       compile=False,
       batch_size=2,
   )
   print(multipoles)

`atom_model_predict` returns charge, dipole, and quadrupole arrays grouped by
molecule when `return_mol_arrays=True`.

Run Pretrained APNet2 Inference
-------------------------------

Use the APNet2 ensemble helper for dimer interaction energies:

.. code-block:: python

   import apnet_pt
   import qcelemental

   mol_dimer = qcelemental.models.Molecule.from_data("""
   0 1
   O 0.000000 0.000000  0.000000
   H 0.758602 0.000000  0.504284
   H 0.260455 0.000000 -0.872893
   --
   0 1
   O 3.000000 0.500000  0.000000
   H 3.758602 0.500000  0.504284
   H 3.260455 0.500000 -0.872893
   """)

   mols = [mol_dimer for _ in range(3)]
   interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
       mols,
       compile=False,
       batch_size=2,
   )
   print(interaction_energies)

The returned NumPy array has shape `(N, 5)` with total energy in column 0,
followed by electrostatics, exchange, induction, and dispersion.

Train a Model
-------------

The training entry point in the README is `train_models.py`. For a short APNet2
run:

.. code-block:: bash

   python3 ./train_models.py \
       --train_ap2 \
       --ap_model_path ./models/example/ap2_example.pt \
       --n_epochs 5

For the atomic multipole model:

.. code-block:: bash

   python3 ./train_models.py \
       --train_am \
       --am_model_path ./models/example/am_example.pt \
       --n_epochs 5

Next Steps
----------

- Read the :doc:`user_guide` for package structure and common workflows.
- Use the :doc:`api` reference to inspect the documented public functions.
- Build the docs locally with `python -m sphinx -b html docs docs/_build/html`.
