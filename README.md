# QCMLForge

[![Tests](https://github.com/Awallace3/QCMLForge/actions/workflows/test.yml/badge.svg)](https://github.com/Awallace3/QCMLForge/actions/workflows/test.yml)

Leverage QCArchive data for creating QC ML models. AP-Net2 has been
re-implemented in PyTorch with newer versions to come.

# PyTorch AP-Net2 
Code re-implemented from TensorFlow version located [here](https://github.com/zachglick/apnet)

## Installation

To install the package, run the following command:
```bash
conda env create -f environment.yml
conda activate qcml
pip install -e .
```
### Common Issues
If you get an OS.Error when running qcml related to torch-scatter, you likely need
to install a specific version through the following example:
```bash
# If you want the CUDA version
pip uninstall torch-geometric torch-scatter
export TORCH=2.7.0
export CUDA=cu126 # for cuda version 12.8
pip install torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# If you want the CPU version
pip uninstall torch-geometric torch-scatter
export TORCH=2.7.0
pip install torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
```

## Usage Workshop Demo
A QCArchive+QCMLForge+Cybershuttle workshop demo is available
[here](https://github.com/Awallace3/psi4_interaction_energy_cybershuttle). This
demonstrates a complete workflow of using QCArchive to generate QM data and
then training AP-Net models with QCMLForge.

## Inference
### AtomModel multipole example
To run the AtomModel inference, run the following command:
```py
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
mols = [mol_mon for _ in range(3)] # example of using multiple molecules
multipoles = apnet_pt.pretrained_models.atom_model_predict(
    mols,
    compile=False,
    batch_size=2,
)
print(multipoles)
# multipoles = [[np.array(q) for q in qs], [[np.array(d) for d in ds], [np.array(qp) for qp in qps]]]
```
### APNet2Model example
To run the APNet2Model inference, run the following command:
```py
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
# interaction_energies = np.array((N, 5)), where [[E_total, E_elst, E_exch, E_ind, E_disp]...]
# [[-1.4542807  -2.25828605  2.25395055 -0.49102123 -0.95892397]
#  [-1.45427967 -2.258285    2.25395055 -0.49102128 -0.95892394]
#  [-1.45428585 -2.25829129  2.25395064 -0.49102129 -0.95892391]]
```

## Training
To train the model, run the following command:
```bash
python3 ./train_models.py \
    --train_ap2 \
    --ap_model_path ./models/example/ap2_example.pt \
    --n_epochs 5 
```

# PyTorch AtomicModule 
Code re-implemented from TensorFlow version located [here](https://github.com/zachglick/apnet)

## Training
To train the model, run the following command:
```bash
python3 ./train_models.py \
    --train_am \
    --am_model_path ./models/example/am_example.pt \
    --n_epochs 5 
```

# Objectives

- [X] Extend AtomMPNN to predict Hirshfeld ratios
- [ ] Add classical induction model for AP3

# Acknowledgements

The free-atom polarizabilities come from
[libmbd](https://github.com/libmbd/libmbd/blob/master/src/pymbd/vdw-params.csv).
To cite Hirshfeld model, please cite libmbd and the original paper to give
appropriate credit for their indirect contributions.
