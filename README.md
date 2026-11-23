# QCMLForge

[![Tests](https://github.com/Awallace3/QCMLForge/actions/workflows/test.yml/badge.svg)](https://github.com/Awallace3/QCMLForge/actions/workflows/test.yml)

Leverage QCArchive data for creating QC ML models. AP-Net2 has been
re-implemented in PyTorch with newer versions to come.

# PyTorch AP-Net2 
Code re-implemented from TensorFlow version located [here](https://github.com/zachglick/apnet)

## Installation

To install the package, run the following commands:
```bash
# --depth 1 skips the multi-GB git history; drop it if you need the history
git clone --depth 1 https://github.com/Awallace3/QCMLForge.git
cd QCMLForge
conda env create -f environment.yml
conda activate qcml
pip install -e .
```

For NVIDIA GPUs, use `environment-gpu.yml` instead. It pins PyTorch's CUDA 12.6
build of torch 2.14 (which still runs on V100s) and includes the CUDA headers
and compilers that `torch.compile` needs on the GPU (driver must support
CUDA >= 12.6):

```bash
conda env create -f environment-gpu.yml
conda activate qcml-gpu
pip install -e . --no-deps
```

### Running tests

Some tests require pretrained model artifacts from the QCMLForge Hugging Face
repository. Enable downloads before running pytest:

```bash
export QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1
python -m pytest tests/
```

Pretrained-model tests are skipped when this environment variable is not
enabled or when their required artifacts cannot be resolved. Tests that do not
need pretrained models still run normally.

Tests marked `gpu` also run on CUDA and are skipped when no GPU is available.

### Common Issues
QCMLForge supports PyTorch 2.14 and 2.15. `environment.yml` (CPU and CI)
installs the 2.15 CPU wheel with pip, since conda-forge does not package 2.15.
`environment-gpu.yml` installs PyTorch's torch 2.14 CUDA 12.6 wheel, because
every torch 2.15 CUDA build targets CUDA 13, which no longer supports V100s
(sm_70). To install by hand:
```bash
# CPU
pip install torch==2.15.* --extra-index-url https://download.pytorch.org/whl/cpu
# CUDA on any GPU, including V100
pip install torch==2.14.1+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
# CUDA 13.2 on sm_75+ GPUs only (RTX 30xx, A100, H100, H200)
pip install torch==2.15.*
```

torch-geometric is pure Python and QCMLForge uses none of its compiled
extensions. If you get an OSError related to torch-geometric, reinstall it
without any extension wheels:
```bash
pip uninstall -y torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-geometric==2.7.0
```

## Usage Workshop Demo
A QCArchive+QCMLForge+Cybershuttle workshop demo is available
[here](https://github.com/Awallace3/psi4_interaction_energy_cybershuttle). This
demonstrates a complete workflow of using QCArchive to generate QM data and
then training AP-Net models with QCMLForge.

## Inference
### Pretrained weights
Pretrained weights are not shipped with the package. They are downloaded from
[Hugging Face](https://huggingface.co/awallace3/qcmlforge) on first use (about
80 MB for the default APNet2 ensemble) and cached in `~/.cache/huggingface`.
Enable the download before running the examples below:

```bash
export QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1
```

Without this variable, a script run from a terminal asks `Download now? [y/N]`,
and a non-interactive run (Jupyter, SLURM/batch jobs, `python script.py <
/dev/null`) fails with `RuntimeError: Missing pretrained model in local cache`.
After the first download the weights load from the cache and the variable is
no longer needed.

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
charges, dipoles, quadrupoles = multipoles
# Each is a list with one array per molecule:
# charges[i].shape == (n_atoms,), dipoles[i].shape == (n_atoms, 3),
# quadrupoles[i].shape == (n_atoms, 3, 3)
print(charges[0])
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
# interaction_energies: np.ndarray of shape (N, 5) in kcal/mol, columns
# [E_total, E_elst, E_exch, E_ind, E_disp]; last digits vary run to run
# [[-2.67262248 -3.41546969  2.33096213 -0.61418158 -0.97393334]
#  [-2.67262368 -3.41547084  2.33096209 -0.61418151 -0.97393341]
#  [-2.67261759 -3.41546469  2.33096199 -0.61418152 -0.97393337]]
```

By default this uses the APNet2 ensemble trained by QCMLForge, which reaches
0.204 kcal/mol total MAE on the paper's 150 000-dimer Splinter test split. Pass
`weights="ap2_tf_paper"` to use the ensemble published with the AP-Net2 paper
instead, converted from TensorFlow and verified against the original
predictions, or `weights="qcmlforge_v1"` for the pre-`AtomMPNN`-fix ensemble
that used to be the default:

```py
interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
    mols,
    compile=False,
    batch_size=2,
    weights="ap2_tf_paper",
)
```

See [docs/apnet2-pretrained-weights.md](docs/apnet2-pretrained-weights.md) for
what each weight set contains and how they score, and
[docs/apnet2-tensorflow-weights.md](docs/apnet2-tensorflow-weights.md) for the
TensorFlow parity data and the loading caveats.

#### F-SAPT-like fragment-pair energies with APNet2

Use `apnet2_model_predict_pairs` to see which groups contribute to the dimer
interaction. It sums APNet2's predicted atom-pair energies for each pair of
groups following the usage of F-SAPT in Psi4 (1-indexed fA/fB fragments).

The [APNet2 ensemble test](tests/test_ensemble.py) uses a 26-atom dimer: two
methyl groups on ethane (A) interact with the peptide and tert-butyl groups on
monomer B. Supply one dictionary per dimer in each of `fAs` and `fBs`. Atom
numbers are **1-based across the whole dimer**: A uses 1–8 and B uses 9–26. The
most useful output is the dataframe named `df_fragments` in the following
example:

```py
from qcelemental.models import Molecule
from apnet_pt.pretrained_models import apnet2_model_predict_pairs


mol_fsapt = Molecule.from_data("""
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com

""")
dimer_energies, atom_pairs, df_fragments = apnet2_model_predict_pairs(
    [mol_fsapt],
    fAs=[
        {
            "Methyl1_A": [1, 2, 7, 8],  # 1-indexed fragment labels!
            "Methyl2_A": [3, 4, 5, 6],  # 1-indexed fragment labels!
        }
    ],
    fBs=[
        {
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        }
    ],
    ap2_fused=True,
    weights="qcmlforge_v1",
    compile=False,
)
print(df_fragments.to_string(index=False))
# prints at the bottom
#              fA-fB     total      elst      exch      indu      disp
#Methyl1_A-Peptide_B  0.154551  0.168962 -0.003577  0.001630 -0.012463
#Methyl1_A-T-Butyl_B -0.676424 -0.423032  0.111288 -0.030383 -0.334297
#Methyl2_A-Peptide_B -0.302270 -0.212438 -0.042306 -0.008193 -0.039332
#Methyl2_A-T-Butyl_B  0.284465 -0.827971  4.198934 -0.515159 -2.571339
```

### ΔAP-Net2 error and compute-time estimates
`dapnet2_model_predict` estimates the interaction-energy error (kcal/mol) of a
level of theory (`m1`) relative to CCSD(T)/CBS/CP (the only `m2`). Models are
downloaded from Hugging Face on first use.

```py
import apnet_pt
import qcelemental
from pprint import pprint as pp

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
mols = [mol_dimer for _ in range(3)] # example of using multiple molecules
# Valid m1 values, e.g. "MP2/aug-cc-pVTZ/CP", "B3LYP-D3/aug-cc-pVTZ/unCP"
pp(apnet_pt.pretrained_models.dapnet2_levels_of_theory_pretrained())

errors = apnet_pt.pretrained_models.dapnet2_model_predict(
    mols, m1="MP2/aug-cc-pVTZ/CP", m2="CCSD(T)/CBS/CP", compile=False
)
print(errors)
# [0.28017625 0.28017625 0.28017625]  (kcal/mol, one per dimer)
```

To tabulate error estimates and estimated Psi4 CPU times (log10 s) across many
methods and basis sets, pass the same dimers to the estimator. It needs `psi4`,
which the main `qcml` environment does not include, so use the estimator
environment:

```bash
conda env create -f environment-estimator.yml
conda activate qcml-estimator
```

If you get `ModuleNotFoundError: No module named 'psi4'`, you are not in the
`qcml-estimator` environment.

```py
# continuing from previous snippet
from qcml_mcp import ie_time_esimator_script as est

df = est.main(mols=mols, methods=["MP2", "wB97X-V"],
              bases=["aug-cc-pVTZ"], using_cp=True, auto_download=True)
# columns: "Level of Theory", "ERROR ESTIMATES (kcal/mol)",
#          "ESTIMATED CPU TIMES (log10(s))"
print(df[["Level of Theory", "ERROR ESTIMATES (kcal/mol)",
          "ESTIMATED CPU TIMES (log10(s))"]].to_string())
#           Level of Theory  ERROR ESTIMATES (kcal/mol)  ESTIMATED CPU TIMES (log10(s))
# 0      MP2/aug-cc-pVTZ/CP                    0.280176                        1.033484
# 0      MP2/aug-cc-pVTZ/CP                    0.280176                        1.033484
# 0      MP2/aug-cc-pVTZ/CP                    0.280176                        1.033484
# 1  wB97X-V/aug-cc-pVTZ/CP                    0.062213                        1.368516
# 1  wB97X-V/aug-cc-pVTZ/CP                    0.062213                        1.368516
# 1  wB97X-V/aug-cc-pVTZ/CP                    0.062213                        1.368516
```

## Training Models
To train models, run the following command:
```bash
python3 ./train_models.py \
    --train_ap2 \
    --ap_model_path ./models/example/ap2_example.pt \
    --n_epochs 5 
```

## Reproducing the original TensorFlow AP-Net2

`models/ap2_tf_paper/` holds the published TensorFlow ensemble converted to PyTorch
checkpoints, which reproduce the original model's predictions to float32
accumulation noise. See
[Running APNet2 with the original TensorFlow weights](docs/apnet2-tensorflow-weights.md).

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

# Acknowledgements

The free-atom polarizabilities come from
[libmbd](https://github.com/libmbd/libmbd/blob/master/src/pymbd/vdw-params.csv).
To cite Hirshfeld model, please cite libmbd and the original paper to give
appropriate credit for their indirect contributions.
