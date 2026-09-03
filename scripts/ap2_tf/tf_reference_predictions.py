#!/usr/bin/env python
"""Record reference AP-Net2 predictions from the original TensorFlow SavedModels.

Runs in the legacy TensorFlow 2.3 / python 3.8 environment against
``github.com/zachglick/apnet`` on the ``sparse`` branch.  It reads the dimer set
written by ``make_parity_dimers.py``, rebuilds each dimer (and each monomer) as
a qcelemental Molecule through ``parity_common``, evaluates every published atom
and pair model, and writes the predictions plus provenance.

The result is committed as a fixture so ``tests/test_ap2_tf_parity.py`` can
assert that the converted PyTorch weights reproduce the original numbers without
needing TensorFlow, python 3.8, or the 100 MB of SavedModels present.

Execution defaults to CPU.  A fixture is only useful if it is reproducible, and
float32 reductions on a V100 do not have to match the CPU bit for bit; pass
``--gpu`` only to check that the GPU path agrees with the recorded CPU numbers.
"""

import os
import sys

import argparse
import hashlib
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

import parity_common

MODEL_INDICES = (0, 1, 2, 3, 4)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(path):
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dimers-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument(
        "--vintage",
        default="",
        choices=["", "_old"],
        help="'' selects atomN/pairN (the vintage the committed PyTorch weights "
        "descend from); '_old' selects the 36-row-embedding atomN_old/pairN_old.",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    for target in (args.out_npz, args.out_manifest):
        if os.path.exists(target) and not args.overwrite:
            raise SystemExit("Refusing to overwrite existing %s" % target)

    import tensorflow as tf

    import apnet
    from apnet import constants
    from apnet.atom_model import AtomModel
    from apnet.pair_model import PairModel

    apnet_dir = os.path.dirname(os.path.dirname(os.path.realpath(apnet.__file__)))
    savedmodel_root = os.path.dirname(os.path.realpath(apnet.__file__))

    if constants.au2ang != parity_common.AU2ANG:
        raise SystemExit(
            "au2ang mismatch: apnet %r vs parity_common %r"
            % (constants.au2ang, parity_common.AU2ANG)
        )

    records, labels = parity_common.load_dimers(args.dimers_npz)
    dimers = []
    monomers_a = []
    monomers_b = []
    round_trip = []
    for record in records:
        dimer = parity_common.build_dimer(record)
        round_trip.append(parity_common.verify_round_trip(record, dimer))
        dimers.append(dimer)
        monomers_a.append(
            parity_common.build_monomer(record["ZA"], record["RA"], record["TQA"])
        )
        monomers_b.append(
            parity_common.build_monomer(record["ZB"], record["RB"], record["TQB"])
        )

    # Confirm apnet's own featuriser sees exactly the elements and Angstrom
    # coordinates the fixture claims, rather than trusting the Molecule round
    # trip in the abstract.
    from apnet.util import qcel_to_dimerdata

    for index, (dimer, record) in enumerate(zip(dimers, records)):
        r_a, r_b, z_a, z_b, _, _ = qcel_to_dimerdata(dimer)
        if not np.array_equal(z_a, record["ZA"]) or not np.array_equal(z_b, record["ZB"]):
            raise SystemExit("dimer %d: element list changed through qcel" % index)
        if not np.array_equal(
            r_a.astype(np.float32), record["RA"].astype(np.float32)
        ) or not np.array_equal(r_b.astype(np.float32), record["RB"].astype(np.float32)):
            raise SystemExit("dimer %d: coordinates changed through qcel" % index)

    arrays = {}
    model_provenance = {}
    for index in MODEL_INDICES:
        atom_dir = os.path.join(
            savedmodel_root, "atom_models", "atom%d%s" % (index, args.vintage)
        )
        pair_dir = os.path.join(
            savedmodel_root, "pair_models", "pair%d%s" % (index, args.vintage)
        )

        atom_model = AtomModel.from_file(atom_dir)
        multipoles_a = atom_model.predict(monomers_a)
        multipoles_b = atom_model.predict(monomers_b)
        arrays["atom%d_multipoles_A" % index] = np.concatenate(
            [np.asarray(m) for m in multipoles_a], axis=0
        ).astype(np.float64)
        arrays["atom%d_multipoles_B" % index] = np.concatenate(
            [np.asarray(m) for m in multipoles_b], axis=0
        ).astype(np.float64)

        pair_model = PairModel.from_file(pair_dir)
        arrays["pair%d_components" % index] = np.asarray(
            pair_model.predict(dimers)
        ).astype(np.float64)

        model_provenance["atom%d" % index] = {
            "savedmodel_dir": atom_dir,
            "saved_model_pb_sha256": sha256_file(os.path.join(atom_dir, "saved_model.pb")),
            "config": {
                k: (float(v) if isinstance(v, float) else v)
                for k, v in atom_model.model.get_config().items()
            },
        }
        model_provenance["pair%d" % index] = {
            "savedmodel_dir": pair_dir,
            "saved_model_pb_sha256": sha256_file(os.path.join(pair_dir, "saved_model.pb")),
            "config": {
                k: (float(v) if isinstance(v, float) else v)
                for k, v in pair_model.model.get_config().items()
            },
        }
        print(
            "%s: pair total MAE vs labels %.6f kcal/mol"
            % (
                "pair%d%s" % (index, args.vintage),
                float(
                    np.mean(
                        np.abs(
                            arrays["pair%d_components" % index].sum(axis=1)
                            - labels.sum(axis=1)
                        )
                    )
                )
                if labels is not None
                else float("nan"),
            ),
            flush=True,
        )

    np.savez(args.out_npz, **arrays)

    manifest = {
        "dimers": {
            "npz": os.path.realpath(args.dimers_npz),
            "sha256": sha256_file(args.dimers_npz),
            "count": len(records),
        },
        "vintage": args.vintage or "new",
        "device": "gpu" if args.gpu else "cpu",
        "au2ang": parity_common.AU2ANG,
        "max_angstrom_round_trip_error": max(round_trip),
        "versions": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "source_repo": git_provenance(apnet_dir),
        "models": model_provenance,
        "outputs": {"npz": {"path": str(args.out_npz), "sha256": sha256_file(args.out_npz)}},
    }
    if labels is not None:
        manifest["reference_total_mae_vs_labels"] = {
            "pair%d" % i: float(
                np.mean(
                    np.abs(arrays["pair%d_components" % i].sum(axis=1) - labels.sum(axis=1))
                )
            )
            for i in MODEL_INDICES
        }
    with open(args.out_manifest, "w") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("wrote %s" % args.out_npz)
    print("  npz sha256 %s" % manifest["outputs"]["npz"]["sha256"])
    print("  manifest   %s" % args.out_manifest)


if __name__ == "__main__":
    main()
