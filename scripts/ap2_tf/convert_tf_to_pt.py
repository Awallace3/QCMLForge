"""Convert exported AP-Net2 TensorFlow weights into QCMLForge PyTorch checkpoints.

Stage two of the conversion. Reads the ``.npz`` + manifest produced by
``export_tf_savedmodel.py`` and writes a ``.pt`` checkpoint in the
``{"model_state_dict": ..., "config": ...}`` layout that
``AtomModel.set_pretrained_model`` and ``APNet2Model.set_pretrained_model``
expect. Needs neither TensorFlow nor the legacy interpreter.

Mapping
-------
``model.variables`` of a subclassed ``tf.keras.Model`` is ordered by attribute
creation in ``__init__``, so the mapping below is positional, and every position
is additionally checked against the variable name recorded in the manifest. A
SavedModel with a different layout therefore fails loudly instead of being
silently mis-mapped.

``KerasPairModel`` creates ``distance_layer_im`` before the readout layers and
``distance_layer`` after them. Both add a weight literally named
``frequencies:0``, so position -- not name -- is what tells them apart. The
embedded, frozen ``KerasAtomModel`` occupies the first 135 positions of a pair
SavedModel; the pair checkpoint does not carry them, because QCMLForge composes
the pair model from a separately loaded ``AtomMPNN``.

Keras ``Dense`` stores ``kernel`` as ``[in, out]`` while ``torch.nn.Linear``
stores ``weight`` as ``[out, in]``, so every kernel is transposed. Biases and
embedding tables transfer unchanged.

Usage:

    python convert_tf_to_pt.py --npz tf_npz/atom0.npz --kind atom --out atom0.pt
    python convert_tf_to_pt.py --npz tf_npz/pair0.npz --kind pair --out pair0.pt
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch

# A FeedForwardLayer is four Dense layers; QCMLForge interleaves activations, so
# the torch Sequential indices are the even numbers.
FF_SUBLAYERS = (0, 2, 4, 6)

# TF scope prefix -> torch ModuleList attribute, for the per-message-pass nets.
ATOM_STACKS = (
    ("charge_update_%d", "charge_update_layers"),
    ("dipole_update_%d", "dipole_update_layers"),
    ("qpole1_update_%d", "qpole1_update_layers"),
    ("qpole2_update_%d", "qpole2_update_layers"),
    ("charge_readout_%d", "charge_readout_layers"),
)
# Bare Dense layers (one kernel + one bias each), not FeedForwardLayers.
ATOM_SINGLE_STACKS = (
    ("dipole_readout_%d", "dipole_readout_layers"),
    ("qpole_readout_%d", "qpole_readout_layers"),
)
# TF readout scope -> torch attribute. Note "ind" vs "indu".
PAIR_READOUTS = (
    ("readout_layer_elst", "readout_layer_elst"),
    ("readout_layer_exch", "readout_layer_exch"),
    ("readout_layer_ind", "readout_layer_indu"),
    ("readout_layer_disp", "readout_layer_disp"),
)
PAIR_STACKS = (
    ("update_layer_%d", "update_layers"),
    ("directional_layer_%d", "directional_layers"),
)

N_ATOM_VARIABLES = 135
N_PAIR_VARIABLES = 218


class Cursor:
    """Walks the exported variables in order, asserting each name as it goes."""

    def __init__(self, arrays, entries):
        self.arrays = arrays
        self.entries = entries
        self.position = 0
        self.state = {}

    def take(self, expect_scope):
        if self.position >= len(self.entries):
            raise ValueError("ran off the end of the variable list")
        entry = self.entries[self.position]
        name = entry["name"]
        if expect_scope is None:
            if not name.endswith("frequencies:0"):
                raise ValueError(
                    "position %d: expected a frequencies weight, found %r"
                    % (self.position, name)
                )
        elif ("/%s/" % expect_scope) not in name:
            raise ValueError(
                "position %d: expected scope %r, found %r"
                % (self.position, expect_scope, name)
            )
        self.position += 1
        return np.asarray(self.arrays[entry["key"]]), name

    def put(self, key, array, transpose=False):
        if key in self.state:
            raise ValueError("duplicate destination key %r" % key)
        if transpose:
            array = array.T
        self.state[key] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def feed_forward(self, tf_scope, pt_prefix):
        """Four Dense layers -> torch Sequential at indices 0, 2, 4, 6."""
        for sublayer in FF_SUBLAYERS:
            kernel, name = self.take(tf_scope)
            if "kernel" not in name:
                raise ValueError("expected a kernel at %r" % name)
            bias, name = self.take(tf_scope)
            if "bias" not in name:
                raise ValueError("expected a bias at %r" % name)
            self.put("%s.%d.weight" % (pt_prefix, sublayer), kernel, transpose=True)
            self.put("%s.%d.bias" % (pt_prefix, sublayer), bias)

    def dense(self, tf_scope, pt_prefix):
        """A single bare Dense layer -> torch Linear."""
        kernel, name = self.take(tf_scope)
        if "kernel" not in name:
            raise ValueError("expected a kernel at %r" % name)
        bias, name = self.take(tf_scope)
        if "bias" not in name:
            raise ValueError("expected a bias at %r" % name)
        self.put("%s.weight" % pt_prefix, kernel, transpose=True)
        self.put("%s.bias" % pt_prefix, bias)

    def frequencies(self, pt_key):
        array, _ = self.take(None)
        self.put(pt_key, array)

    def embedding(self, pt_key):
        entry = self.entries[self.position]
        if "embedding" not in entry["name"]:
            raise ValueError(
                "position %d: expected an embedding, found %r"
                % (self.position, entry["name"])
            )
        self.position += 1
        self.put(pt_key, np.asarray(self.arrays[entry["key"]]))


def convert_atom(cursor, n_message):
    cursor.frequencies("distance_layer.frequencies")
    cursor.embedding("embed_layer.weight")
    cursor.embedding("guess_layer.weight")
    for scope_fmt, pt_attr in ATOM_STACKS:
        for i in range(n_message):
            cursor.feed_forward(scope_fmt % i, "%s.%d" % (pt_attr, i))
    for scope_fmt, pt_attr in ATOM_SINGLE_STACKS:
        for i in range(n_message):
            cursor.dense(scope_fmt % i, "%s.%d" % (pt_attr, i))


def convert_pair(cursor, n_message):
    # Skip the embedded, frozen atom model; QCMLForge supplies it separately.
    cursor.position = N_ATOM_VARIABLES
    cursor.frequencies("distance_layer_im.frequencies")
    cursor.embedding("embed_layer.weight")
    for tf_scope, pt_attr in PAIR_READOUTS:
        cursor.feed_forward(tf_scope, pt_attr)
    cursor.frequencies("distance_layer.frequencies")
    for scope_fmt, pt_attr in PAIR_STACKS:
        for i in range(n_message):
            cursor.feed_forward(scope_fmt % i, "%s.%d" % (pt_attr, i))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="npz written by export_tf_savedmodel.py")
    parser.add_argument("--manifest", help="manifest path (default: <npz stem>.manifest.json)")
    parser.add_argument("--kind", required=True, choices=("atom", "pair"))
    parser.add_argument("--out", required=True, help="destination .pt checkpoint")
    parser.add_argument("--r-cut", type=float, default=5.0,
                        help="intramonomer cutoff; a tf.constant in apnet, not a variable")
    parser.add_argument("--r-cut-im", type=float, default=8.0,
                        help="intermonomer cutoff (pair models only)")
    parser.add_argument("--quadrupole-scale", type=float, default=1.5,
                        help="pair-model multipole-electrostatics quadrupole "
                             "prefactor; 1.5 reproduces KerasPairModel.mtp_elst")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit("refusing to overwrite %s (pass --overwrite)" % args.out)

    manifest_path = args.manifest or (os.path.splitext(args.npz)[0] + ".manifest.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    entries = sorted(manifest["variables"], key=lambda e: e["index"])

    expected = N_ATOM_VARIABLES if args.kind == "atom" else N_PAIR_VARIABLES
    if len(entries) != expected:
        raise SystemExit(
            "%s SavedModel should have %d variables, manifest has %d"
            % (args.kind, expected, len(entries))
        )

    expected_npz_sha256 = manifest.get("npz_sha256")
    actual_npz_sha256 = sha256_file(args.npz)
    if expected_npz_sha256 != actual_npz_sha256:
        raise SystemExit(
            "NPZ does not match manifest: expected %s, got %s"
            % (expected_npz_sha256, actual_npz_sha256)
        )

    arrays = np.load(args.npz)
    cursor = Cursor(arrays, entries)

    n_rbf = int(entries[0]["shape"][0])
    n_message = 3
    if args.kind == "atom":
        n_embed = int(entries[1]["shape"][1])
        convert_atom(cursor, n_message)
    else:
        n_embed = int(entries[N_ATOM_VARIABLES + 1]["shape"][1])
        convert_pair(cursor, n_message)

    if cursor.position != len(entries):
        raise SystemExit(
            "consumed %d of %d variables; mapping is incomplete"
            % (cursor.position, len(entries))
        )

    n_neuron = cursor.state[
        "charge_update_layers.0.0.weight" if args.kind == "atom" else "update_layers.0.0.weight"
    ].shape[0] // 2

    config = {
        "n_message": n_message,
        "n_rbf": n_rbf,
        "n_neuron": n_neuron,
        "n_embed": n_embed,
        "r_cut": args.r_cut,
    }
    if args.kind == "pair":
        config["r_cut_im"] = args.r_cut_im
        # ``KerasPairModel.mtp_elst`` multiplies both quadrupole tensors by 3/2
        # before contracting them with T2; QCMLForge exposes that factor as
        # ``quadrupole_scale`` and defaults it to 1.0, so reproducing the
        # original numbers requires carrying 1.5 in the checkpoint config.
        config["quadrupole_scale"] = args.quadrupole_scale

    provenance = {
        "converter": os.path.basename(__file__),
        "kind": args.kind,
        "npz": os.path.basename(args.npz),
        "npz_sha256": actual_npz_sha256,
        "savedmodel_dir": manifest.get("savedmodel_dir"),
        "savedmodel_name": manifest.get("savedmodel_name"),
        "saved_model_pb_sha256": manifest.get("saved_model_pb_sha256"),
        "source_repo": manifest.get("source_repo"),
        "tensorflow": manifest.get("tensorflow"),
        "n_tf_variables": len(entries),
        "n_tf_parameters": manifest.get("n_parameters"),
        "n_pt_tensors": len(cursor.state),
        "n_pt_parameters": int(sum(t.numel() for t in cursor.state.values())),
        "r_cut": args.r_cut,
        "r_cut_im": args.r_cut_im if args.kind == "pair" else None,
        "quadrupole_scale": args.quadrupole_scale if args.kind == "pair" else None,
    }

    torch.save(
        {"model_state_dict": cursor.state, "config": config, "tf_provenance": provenance},
        args.out,
    )
    print(
        "wrote %s: %d tensors, %d parameters (from %d TF variables / %d parameters)"
        % (
            args.out,
            provenance["n_pt_tensors"],
            provenance["n_pt_parameters"],
            provenance["n_tf_variables"],
            provenance["n_tf_parameters"],
        )
    )
    print("  sha256 %s" % sha256_file(args.out))


if __name__ == "__main__":
    main()
