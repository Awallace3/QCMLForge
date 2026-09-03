#!/bin/bash
# Build the legacy environment that runs the original TensorFlow AP-Net2.
#
# The published SavedModels in github.com/zachglick/apnet (branch `sparse`)
# cannot be loaded by any currently supported TensorFlow: apnet/setup.py pins
# `tensorflow>=2.2,<2.4`, and TF 2.3 is the last release whose SavedModel reader
# accepts them.  TF 2.3 wheels exist only for python <= 3.8.  Everything else
# here follows from those two facts.
#
# This environment is needed to (a) export the SavedModel variables for
# scripts/ap2_tf/convert_tf_to_pt.py and (b) regenerate the TensorFlow reference
# predictions in tests/dataset_data/ap2_tf_parity/.  Ordinary use of the
# converted checkpoints in models/ap2_tf/** needs none of it.
#
# Every version below is pinned because pip's resolver does not backtrack far
# enough on this stack to find a working combination on its own:
#
#   numpy==1.18.5     TF 2.3 was built against the 1.18 C API; numpy >= 1.20
#                     breaks it, and numpy >= 1.24 removed the `np.object`
#                     aliases that TF 2.3's python layer still uses.
#   scipy==1.5.4      last release with a python 3.8 wheel that does not
#                     require numpy >= 1.19.
#   pandas==1.1.5     apnet.util reads .pkl frames written by this vintage.
#   protobuf==3.19.6  4.x rejects the SavedModel descriptors TF 2.3 generates.
#   h5py==2.10.0      TF 2.3 calls the h5py 2.x API for .h5 weights.
#   gast==0.3.3       TF 2.3's autograph pins this exact version.
#   pydantic==1.10.21 qcelemental 0.25 is pydantic v1 only.
#   qcelemental==0.25.1  last release supporting python 3.8; supplies the
#                     bohr->angstrom factor apnet.constants re-exports, which
#                     must be the same constant the PyTorch side uses.
#   wandb==0.16.6     last release supporting python 3.8.
#
# cudatoolkit 10.1 / cudnn 7.6 is the CUDA ABI TF 2.3 links against.  Whether a
# given GPU can run it is a separate question: a V100 (sm_70) is within TF
# 2.3's compiled compute capabilities, but newer cards are not.  The reference
# predictions were generated on CPU deliberately -- a fixture has to be
# reproducible, and float32 reduction order differs between CPU and GPU.
#
# Built on a Phoenix login node, not under SLURM: no GPU is required to build,
# and the conda solve is long enough to be inconvenient inside a job.

set -euo pipefail

PREFIX="${PREFIX:-/storage/project/r-cs207-0/awallace43/conda-envs/apnet-tf-legacy}"
CONDA="${CONDA:-/storage/project/r-cs207-0/awallace43/miniconda/bin/conda}"
APNET_SRC="${APNET_SRC:-/storage/project/r-cs207-0/awallace43/gits/apnet}"
APNET_COMMIT="${APNET_COMMIT:-f093e00bf64190ac30a7706d2a90e66871347b76}"

if [ -d "$PREFIX" ]; then
  echo "PREFIX already exists: $PREFIX -- refusing to modify."
  echo "Remove it deliberately if a rebuild is wanted."
  exit 3
fi

if [ ! -d "$APNET_SRC/.git" ]; then
  echo "Expected a git checkout of github.com/zachglick/apnet at $APNET_SRC"
  exit 4
fi

# The SavedModels only exist on `sparse`.  `master` is AP-Net v1 -- a different
# architecture (ACSF/APSF symmetry functions, float64) whose weights have no
# counterpart in QCMLForge.
HAVE=$(git -C "$APNET_SRC" rev-parse HEAD)
if [ "$HAVE" != "$APNET_COMMIT" ]; then
  echo "apnet checkout is at $HAVE, expected $APNET_COMMIT (branch sparse, tag v0.1.0)"
  exit 5
fi

echo "=== [1/5] conda create python=3.8 ==="
"$CONDA" create -y -p "$PREFIX" python=3.8

echo "=== [2/5] cudatoolkit 10.1 / cudnn 7.6 (TF 2.3 CUDA ABI) ==="
"$CONDA" install -y -p "$PREFIX" -c conda-forge cudatoolkit=10.1 cudnn=7.6

PIP="$PREFIX/bin/pip"
echo "=== [3/5] pip install pinned legacy stack ==="
"$PIP" install --no-cache-dir \
  "tensorflow==2.3.4" \
  "numpy==1.18.5" \
  "scipy==1.5.4" \
  "pandas==1.1.5" \
  "protobuf==3.19.6" \
  "h5py==2.10.0" \
  "gast==0.3.3" \
  "pydantic==1.10.21" \
  "qcelemental==0.25.1" \
  "wandb==0.16.6"

echo "=== [4/5] pip install -e apnet (sparse) ==="
"$PIP" install --no-cache-dir --no-deps -e "$APNET_SRC"

echo "=== [5/5] verify ==="
"$PREFIX/bin/python" - <<'PY'
import sys

print("python", sys.version.split()[0])
for name in ("numpy", "scipy", "pandas", "h5py", "pydantic",
             "qcelemental", "wandb", "tensorflow"):
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001 -- report, do not mask
        print(name, "IMPORT FAILED:", type(exc).__name__, exc)
        sys.exit(1)

# Loading one SavedModel is the only check that actually matters; the pins above
# exist to make this line work.
import apnet
from apnet import constants  # noqa: F401
import os
import tensorflow as tf

root = os.path.dirname(apnet.__file__)
loaded = 0
for kind, count in (("atom_models/atom", 5), ("pair_models/pair", 5)):
    for index in range(count):
        tf.keras.models.load_model(os.path.join(root, "%s%d" % (kind, index)))
        loaded += 1
print("loaded %d SavedModels" % loaded)
PY
echo "=== DONE ==="
echo "Reference freeze: scripts/ap2_tf/legacy-tf-env.pip-freeze.txt"
