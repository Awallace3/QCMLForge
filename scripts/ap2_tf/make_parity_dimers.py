#!/usr/bin/env python
"""Extract a small, fixed dimer set for the TensorFlow <-> PyTorch parity check.

Runs in the PyTorch environment, because the processed shards are ``torch.save``
payloads.  The output is a plain npz plus a JSON manifest, so the legacy
TensorFlow 2.3 environment -- which has no torch -- can consume it.

Geometries are widened to float64.  The shards hold float32 and every consumer
casts back to float32, so the widening is exact in both directions and avoids
depending on how any particular numpy version reprs a float32 scalar.

The dimers are taken in shard order, which makes the selection a pure function
of ``--processed-dir``, ``--prefix``, and ``--samples``; ``row_provenance`` in
the manifest names the shard and in-shard index of every row so the choice stays
auditable after the fact.

``--ensure-monatomic`` extends that rule by one case.  A monomer with a single
atom has no intramonomer edge, which is a distinct code path in both
featurisers and the one the original fixture happened to miss entirely -- the
atom model corrupted the multipoles of every atom batched after such a monomer
and 24 dimers of ordinary organic pairs could not see it.  When the first
``samples`` dimers contain no monatomic monomer, two more are appended: the
first dimer in shard order that has one, and the dimer immediately after it.
The trailing dimer is what makes the check bite, because the damage lands on
whatever shares the batch *after* the lone atom, so a monatomic dimer appended
alone at the end of the list would prove nothing.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

COMPONENT_COLUMNS = ("Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug")


def natural_key(text):
    """Order ``..._2_9.pt`` before ``..._2_10.pt``, matching the dataset loader."""

    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", text)]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(data):
    y = data.y.reshape(-1).to(torch.float64).numpy()
    if y.shape != (4,):
        raise SystemExit("y shape %s, expected (4,)" % (y.shape,))
    return {
        "ZA": data.ZA.to(torch.int64).numpy(),
        "ZB": data.ZB.to(torch.int64).numpy(),
        "RA": data.RA.to(torch.float64).numpy(),
        "RB": data.RB.to(torch.float64).numpy(),
        "TQA": int(data.total_charge_A.item()),
        "TQB": int(data.total_charge_B.item()),
        "labels": y,
    }


def is_monatomic(record):
    return len(record["ZA"]) == 1 or len(record["ZB"]) == 1


def collect(processed_dir, prefix, samples, ensure_monatomic=False):
    paths = sorted(
        Path(processed_dir).glob(prefix + "*.pt"), key=lambda p: natural_key(p.name)
    )
    if not paths:
        raise SystemExit("No shards matching %s*.pt in %s" % (prefix, processed_dir))

    records = []
    provenance = []
    shards_read = []
    # ``wanted`` grows once past ``samples``: to +1 when the first monatomic
    # dimer is found and to +2 so the dimer after it comes along to receive any
    # corruption.  ``hunting`` stops the scan as soon as that is satisfied.
    wanted = samples
    hunting = ensure_monatomic

    def take(path, position, record, reason):
        records.append(record)
        provenance.append(
            {"shard": path.name, "index_in_shard": position, "reason": reason}
        )

    for path in paths:
        objects = torch.load(str(path), map_location="cpu", weights_only=False)
        shards_read.append(path.name)
        for position, data in enumerate(objects):
            if len(records) >= wanted and not hunting:
                break
            record = _record(data)
            if len(records) < samples:
                take(path, position, record, "leading")
                if hunting and is_monatomic(record):
                    hunting = False
            elif len(records) < wanted:
                take(path, position, record, "follows monatomic")
                hunting = False
            elif is_monatomic(record):
                wanted = len(records) + 2
                take(path, position, record, "monatomic")
        if len(records) >= wanted and not hunting:
            break

    if len(records) < samples:
        raise SystemExit(
            "Collected %d dimers from %d shards, wanted %d"
            % (len(records), len(paths), samples)
        )
    if hunting:
        raise SystemExit(
            "No dimer with a monatomic monomer, followed by another dimer, "
            "exists in %s%s*.pt" % (processed_dir, prefix)
        )
    return records, provenance, shards_read


def flatten(records):
    return {
        "sizes_A": np.array([len(r["ZA"]) for r in records], dtype=np.int64),
        "sizes_B": np.array([len(r["ZB"]) for r in records], dtype=np.int64),
        "ZA": np.concatenate([r["ZA"] for r in records]).astype(np.int64),
        "ZB": np.concatenate([r["ZB"] for r in records]).astype(np.int64),
        "RA": np.concatenate([r["RA"] for r in records]).astype(np.float64),
        "RB": np.concatenate([r["RB"] for r in records]).astype(np.float64),
        "TQA": np.array([r["TQA"] for r in records], dtype=np.int64),
        "TQB": np.array([r["TQB"] for r in records], dtype=np.int64),
        "labels": np.stack([r["labels"] for r in records]).astype(np.float64),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--prefix", required=True, help="e.g. dimer_ap2_fused_test_spec_2_")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument(
        "--ensure-monatomic",
        action="store_true",
        help="append the first monatomic-monomer dimer and its successor when "
        "the leading `samples` dimers contain none",
    )
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for target in (args.out_npz, args.out_manifest):
        if Path(target).exists() and not args.overwrite:
            raise SystemExit("Refusing to overwrite existing %s" % target)

    records, provenance, shards_read = collect(
        args.processed_dir, args.prefix, args.samples, args.ensure_monatomic
    )
    arrays = flatten(records)
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **arrays)

    elements = sorted(set(arrays["ZA"].tolist()) | set(arrays["ZB"].tolist()))
    manifest = {
        "source": {
            "processed_dir": str(args.processed_dir),
            "prefix": args.prefix,
            "shards_read": shards_read,
            "selection_rule": (
                "first `samples` dimers in naturally sorted shard order"
                + (
                    ", plus the first dimer with a monatomic monomer and its "
                    "successor when none of those has one"
                    if args.ensure_monatomic
                    else ""
                )
            ),
        },
        "samples": len(records),
        "atoms_A_total": int(arrays["sizes_A"].sum()),
        "atoms_B_total": int(arrays["sizes_B"].sum()),
        "elements_present": [int(z) for z in elements],
        "monatomic_monomers": int(
            np.count_nonzero(arrays["sizes_A"] == 1)
            + np.count_nonzero(arrays["sizes_B"] == 1)
        ),
        "charged_monomers": int(
            np.count_nonzero(arrays["TQA"]) + np.count_nonzero(arrays["TQB"])
        ),
        "label_means": {
            c: float(arrays["labels"][:, i].mean())
            for i, c in enumerate(COMPONENT_COLUMNS)
        },
        "outputs": {"npz": {"path": str(args.out_npz), "sha256": sha256_file(args.out_npz)}},
        "row_provenance": provenance,
    }
    with open(str(args.out_manifest), "w") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")

    print("wrote %s (%d dimers)" % (args.out_npz, len(records)))
    print("  npz sha256 %s" % manifest["outputs"]["npz"]["sha256"])
    print("  elements   %s" % elements)
    print("  manifest   %s" % args.out_manifest)


if __name__ == "__main__":
    main()
