"""Export the variables of an original AP-Net2 TensorFlow SavedModel to ``.npz``.

This is stage one of the TensorFlow -> PyTorch weight conversion. It runs inside
a legacy environment (``tensorflow>=2.2,<2.4``, python 3.8) that can read the
SavedModels published in https://github.com/zachglick/apnet on the ``sparse``
branch. It writes plain arrays plus a manifest, so stage two
(``convert_tf_to_pt.py``) needs neither TensorFlow nor the legacy interpreter.

Variables are emitted in ``model.variables`` order, which for a subclassed
``tf.keras.Model`` follows the order the attributes were created in
``__init__``. That order is what makes the two identically named
``frequencies:0`` variables of ``KerasPairModel`` distinguishable, so the
manifest records the index of every variable and never relies on names alone.

Usage (legacy env):

    python export_tf_savedmodel.py --out-dir tf_npz \
        /path/to/apnet/apnet/atom_models/atom0 \
        /path/to/apnet/apnet/pair_models/pair0
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(path):
    """Best-effort git description of the repository holding ``path``."""
    def run(*args):
        try:
            out = subprocess.check_output(
                ("git", "-C", path) + args, stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        return out.decode().strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--tags", "--always"),
        "dirty": bool(run("status", "--porcelain")),
        "remote": run("config", "--get", "remote.origin.url"),
    }


def export(savedmodel_dir, out_dir):
    name = os.path.basename(savedmodel_dir.rstrip("/"))
    npz_path = os.path.join(out_dir, name + ".npz")
    manifest_path = os.path.join(out_dir, name + ".manifest.json")
    for path in (npz_path, manifest_path):
        if os.path.exists(path):
            raise SystemExit("refusing to overwrite %s" % path)

    model = tf.keras.models.load_model(savedmodel_dir)

    arrays = {}
    entries = []
    for index, var in enumerate(model.variables):
        arr = np.asarray(var.numpy())
        key = "var_%03d" % index
        arrays[key] = arr
        entries.append(
            {
                "index": index,
                "key": key,
                "name": var.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size": int(arr.size),
                "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
            }
        )

    np.savez(npz_path, **arrays)

    pb_path = os.path.join(savedmodel_dir, "saved_model.pb")
    manifest = {
        "savedmodel_dir": os.path.realpath(savedmodel_dir),
        "savedmodel_name": name,
        "saved_model_pb_sha256": sha256_file(pb_path) if os.path.exists(pb_path) else None,
        "source_repo": git_provenance(savedmodel_dir),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "n_variables": len(entries),
        "n_parameters": int(sum(e["size"] for e in entries)),
        "variables": entries,
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    manifest["npz_sha256"] = sha256_file(npz_path)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        "exported %s: %d variables, %d parameters -> %s"
        % (name, len(entries), manifest["n_parameters"], npz_path)
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("savedmodels", nargs="+", help="SavedModel directories")
    parser.add_argument("--out-dir", required=True, help="destination for .npz + manifests")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for savedmodel_dir in args.savedmodels:
        export(savedmodel_dir, args.out_dir)


if __name__ == "__main__":
    main()
