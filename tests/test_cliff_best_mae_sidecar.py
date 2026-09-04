"""Tests for the best-MAE sidecar on the CLIFF classical trainer.

The primary checkpoint is starred on validation *MSE*, but the S66x8 gate and
every per-component table read these models in *MAE*, and the two selectors
disagree.  The l<=2 exchange arm (job 12800472) last starred chunk-local epoch
3 of 11 while validation exchange kept improving from 0.728 to 0.724 through
epoch 10; with no sidecar those seven epochs were unrecoverable.

The contract tested here is that the sidecar is purely *additive*: it writes a
second checkpoint and a record beside the primary artifact, and changes neither
the primary checkpoint, the live model's device, nor the selector that produced
it.
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt import model_io  # noqa: E402
from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402

HEAD_KWARGS = dict(n_message=1, n_neuron=8, n_embed=4)


def _harness(nested):
    return mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        **HEAD_KWARGS,
    )


# ------------------------------------------------------------------ the paths


def test_sidecar_paths_replace_the_extension():
    ckpt, record = model_io.best_mae_sidecar_paths("/models/run/cliff2.pt")
    assert ckpt == "/models/run/cliff2.best-mae.pt"
    assert record == "/models/run/cliff2.best-mae.json"


# ------------------------------------------------------------------ the floor
#
# Every way the floor can be wrong has to end as "no banked best", which is
# exactly the single-run behaviour.  The failure it exists to stop is a later
# chunk overwriting an earlier chunk's banked epoch with a worse one, because
# each chunk seeds its selector from its own fresh pre-training eval.


def test_floor_is_infinite_with_no_path_at_all():
    assert model_io.best_mae_sidecar_floor(None) == float("inf")


def test_floor_is_infinite_when_no_sidecar_exists(tmp_path):
    assert model_io.best_mae_sidecar_floor(str(tmp_path / "cliff2.pt")) == float("inf")


def test_floor_reads_a_previous_chunks_record(tmp_path):
    target = str(tmp_path / "cliff2.pt")
    ckpt, record = model_io.best_mae_sidecar_paths(target)
    Path(ckpt).write_bytes(b"weights")
    model_io.save_best_mae_record(
        record,
        model_save_path=target,
        checkpoint=ckpt,
        val_total_MAE=1.621,
        component_MAE=[0.561, 0.724, 0.336],
        epoch=10,
    )
    assert model_io.best_mae_sidecar_floor(target) == pytest.approx(1.621)


def test_floor_ignores_a_record_written_for_another_path(tmp_path):
    """A record copied beside a different run must not silently gate it."""
    target = str(tmp_path / "cliff2.pt")
    ckpt, record = model_io.best_mae_sidecar_paths(target)
    Path(ckpt).write_bytes(b"weights")
    model_io.save_best_mae_record(
        record,
        model_save_path=str(tmp_path / "somebody_elses.pt"),
        checkpoint=ckpt,
        val_total_MAE=0.1,
        component_MAE=[0.1],
        epoch=3,
    )
    assert model_io.best_mae_sidecar_floor(target) == float("inf")


def test_floor_ignores_a_corrupt_record(tmp_path):
    target = str(tmp_path / "cliff2.pt")
    ckpt, record = model_io.best_mae_sidecar_paths(target)
    Path(ckpt).write_bytes(b"weights")
    Path(record).write_text("{not json")
    assert model_io.best_mae_sidecar_floor(target) == float("inf")


def test_floor_ignores_a_record_whose_checkpoint_vanished(tmp_path):
    """The record alone is not a bank -- the weights are the deliverable."""
    target = str(tmp_path / "cliff2.pt")
    _, record = model_io.best_mae_sidecar_paths(target)
    model_io.save_best_mae_record(
        record,
        model_save_path=target,
        checkpoint=str(tmp_path / "cliff2.best-mae.pt"),
        val_total_MAE=0.4,
        component_MAE=[0.4],
        epoch=1,
    )
    assert model_io.best_mae_sidecar_floor(target) == float("inf")


def test_the_record_write_is_atomic_and_leaves_no_temporary(tmp_path):
    target = str(tmp_path / "cliff2.pt")
    _, record = model_io.best_mae_sidecar_paths(target)
    model_io.save_best_mae_record(
        record,
        model_save_path=target,
        checkpoint=str(tmp_path / "cliff2.best-mae.pt"),
        val_total_MAE=1.0,
        component_MAE=[1.0],
        epoch=0,
    )
    assert not os.path.exists(record + ".tmp")


def test_the_record_epoch_is_labelled_global(tmp_path):
    """The earlier sidecar shipped a chunk-local `epoch`, which cannot be
    compared across a chain.  This one records the trainer's global counter and
    says so, so no table can mislabel which epoch was scored."""
    target = str(tmp_path / "cliff2.pt")
    _, record = model_io.best_mae_sidecar_paths(target)
    model_io.save_best_mae_record(
        record,
        model_save_path=target,
        checkpoint=str(tmp_path / "cliff2.best-mae.pt"),
        val_total_MAE=1.0,
        component_MAE=[1.0],
        epoch=7,
    )
    payload = json.loads(Path(record).read_text())
    assert payload["epoch"] == 7
    assert payload["epoch_is_global"] is True
    assert payload["selector"] == "val_total_MAE"


# ----------------------------------------------------------------- the writer


def test_the_sidecar_writes_a_loadable_checkpoint_and_record(
    nested_hfvr_vw_model, tmp_path
):
    harness = _harness(nested_hfvr_vw_model)
    harness.model_save_path = str(tmp_path / "cliff2.pt")
    harness._save_best_mae_sidecar(
        val_total_MAE=1.621,
        component_MAE=[0.561, 0.724, 0.336],
        epoch=10,
        world_size=1,
        rank_device=torch.device("cpu"),
    )
    ckpt_path, record_path = model_io.best_mae_sidecar_paths(harness.model_save_path)
    assert os.path.exists(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    assert metadata["selector"] == "val_total_MAE"
    assert metadata["val_total_MAE"] == pytest.approx(1.621)
    assert metadata["component_MAE"] == pytest.approx([0.561, 0.724, 0.336])
    assert metadata["epoch"] == 10
    assert metadata["epoch_is_global"] is True

    record = json.loads(Path(record_path).read_text())
    assert record["checkpoint"] == ckpt_path
    assert record["model_save_path"] == harness.model_save_path
    # And it round-trips: the floor a next chunk would read is this value.
    assert model_io.best_mae_sidecar_floor(harness.model_save_path) == pytest.approx(
        1.621
    )


def test_the_sidecar_does_not_write_or_touch_the_primary_checkpoint(
    nested_hfvr_vw_model, tmp_path
):
    """The whole point is that the MSE-selected deliverable is unchanged."""
    harness = _harness(nested_hfvr_vw_model)
    harness.model_save_path = str(tmp_path / "cliff2.pt")
    Path(harness.model_save_path).write_bytes(b"the primary checkpoint")
    harness._save_best_mae_sidecar(
        val_total_MAE=1.0,
        component_MAE=[0.4, 0.4, 0.2],
        epoch=2,
        world_size=1,
        rank_device=torch.device("cpu"),
    )
    assert Path(harness.model_save_path).read_bytes() == b"the primary checkpoint"


def test_the_sidecar_leaves_the_live_model_on_its_device(
    nested_hfvr_vw_model, tmp_path
):
    """`_create_checkpoint` needs CPU weights, so the writer moves the model and
    must move it back -- otherwise the next epoch trains on the wrong device."""
    harness = _harness(nested_hfvr_vw_model)
    harness.model_save_path = str(tmp_path / "cliff2.pt")
    before = {k: v.clone() for k, v in harness.model.state_dict().items()}
    harness._save_best_mae_sidecar(
        val_total_MAE=1.0,
        component_MAE=[1.0],
        epoch=0,
        world_size=1,
        rank_device=torch.device("cpu"),
    )
    after = harness.model.state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key
        assert after[key].device == value.device, key


# ------------------------------------------------------------- the epoch hook


def test_the_epoch_loop_selects_on_summed_component_mae():
    """A structural check on the wiring, which has no cheap end-to-end path.

    The selector must be the *sum* over components of the validation MAE
    vector, and the branch must sit after the primary save so it cannot
    reorder it.
    """
    source = Path(
        REPO_ROOT / "src/apnet_pt/AtomPairwiseModels/mtp_mtp.py"
    ).read_text()
    assert "lowest_val_total_MAE = model_io.best_mae_sidecar_floor(" in source
    assert "val_total_MAE = float(sum(component_MAE_v))" in source
    assert source.index("model_io.save_checkpoint(checkpoint, self.model_save_path)") < (
        source.index("if val_total_MAE < lowest_val_total_MAE:")
    )
