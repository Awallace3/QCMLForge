"""Growing an on-disk fused store to the size a run actually asked for.

``processed_file_names`` is built by globbing the shards that already exist and
then only ever truncating that list down to ``max_size``.  PyG decides whether
to call ``process()`` by checking that every listed processed file exists -- a
check that list satisfies by construction.  So once any shard was on disk the
store could never grow, and a run that asked for 1.5M dimers trained on the
100k that happened to be there without printing a word about it.

These tests pin the decision logic that fixes that: when to grow the store,
when to refuse to rescan a source that has nothing more to give, and when to
say out loud that the store is smaller than requested.
"""

import json
import os
import os.path as osp

import pytest
import torch

from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_module_dataset


SHARD_OBJECTS = 16


def _store(tmp_path, *, shards, max_size, split="train", storage_type="pt"):
    """A dataset instance with ``shards`` shards on disk and PyG's init bypassed.

    The constructor loads an atom model and would process raw data; none of
    that is involved in deciding whether the store is big enough, which is the
    only thing under test here.
    """
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    ds = object.__new__(ap2_fused_module_dataset)
    ds.root = str(tmp_path)
    ds.spec_type = 2
    ds.split = split
    ds.split_db = True
    ds.storage_type = storage_type
    ds.datapoint_storage_n_objects = SHARD_OBJECTS
    ds.points_per_file = SHARD_OBJECTS
    ds.MAX_SIZE = max_size
    ds.in_memory = False
    ds.force_reprocess = False
    ds.print_level = 0
    for idx in range(shards):
        torch.save(
            [torch.zeros(1)] * SHARD_OBJECTS,
            processed / f"dimer_ap2_fused_{split}_spec_2_{idx}{ds.file_extension}",
        )
    return ds


def test_a_store_that_already_covers_max_size_is_not_rebuilt(tmp_path, recwarn):
    ds = _store(tmp_path, shards=6250, max_size=100_000)
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is False
    ds._warn_if_store_is_smaller_than_requested()
    assert [w for w in recwarn] == []


def test_a_store_short_of_max_size_asks_to_be_extended(tmp_path):
    """The exact shape of the bug: 6250 shards on disk, 1.5M dimers requested."""
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True


def test_an_empty_store_is_left_to_the_existing_missing_marker(tmp_path):
    ds = _store(tmp_path, shards=0, max_size=1_500_000)
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is False


def test_max_size_none_never_triggers_a_rebuild(tmp_path, recwarn):
    ds = _store(tmp_path, shards=6250, max_size=None)
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is False
    ds._warn_if_store_is_smaller_than_requested()
    assert [w for w in recwarn] == []


def test_a_source_recorded_as_exhausted_is_not_rescanned(tmp_path):
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=True)
    ds.force_reprocess = False
    with pytest.warns(UserWarning, match="cannot produce more"):
        ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is False


def test_a_record_that_does_not_claim_exhaustion_does_not_block_extension(tmp_path):
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=False)
    ds.force_reprocess = False
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True


def test_a_source_that_since_grew_past_the_record_is_extended_again(tmp_path):
    """An exhaustion record is about a source, not a ceiling on the store."""
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=True)
    with open(ds._store_extent_path()) as fh:
        recorded = json.load(fh)
    recorded["shards"] = 90_000
    with open(ds._store_extent_path(), "w") as fh:
        json.dump(recorded, fh)
    ds.force_reprocess = False
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True


def test_an_unreadable_record_does_not_block_extension(tmp_path):
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    with open(ds._store_extent_path(), "w") as fh:
        fh.write("{not json")
    with pytest.warns(UserWarning, match="unreadable store extent"):
        ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True


def test_a_short_store_reports_the_shortfall(tmp_path):
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    with pytest.warns(UserWarning) as caught:
        ds._warn_if_store_is_smaller_than_requested()
    message = str(caught[0].message)
    assert "100000 dimers" in message
    assert "1500000" in message


def test_the_extent_record_round_trips(tmp_path):
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=True)
    assert ds._recorded_source_extent() == 6250
    with open(ds._store_extent_path()) as fh:
        recorded = json.load(fh)
    assert recorded["requested_max_size"] == 1_500_000
    assert recorded["datapoint_storage_n_objects"] == SHARD_OBJECTS


def test_the_extent_record_write_leaves_no_temporary_file(tmp_path):
    ds = _store(tmp_path, shards=1, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=True)
    leftovers = [
        name
        for name in os.listdir(osp.join(str(tmp_path), "processed"))
        if name.endswith(".tmp")
    ]
    assert leftovers == []


def test_the_extent_record_is_not_mistaken_for_a_shard(tmp_path):
    """The record lives in the same directory the shard glob sweeps."""
    ds = _store(tmp_path, shards=6250, max_size=1_500_000)
    ds._record_store_extent(source_exhausted=True)
    assert len(ds._existing_shard_paths()) == 6250
    assert len(ds.reprocess_file_names()) == 6250


def test_the_shard_glob_agrees_with_reprocess_file_names(tmp_path):
    """Two globs for "what is present" that could drift apart, pinned together."""
    ds = _store(tmp_path, shards=64, max_size=None)
    assert [osp.basename(p) for p in ds._existing_shard_paths()] == list(
        ds.reprocess_file_names()
    )


def test_shards_are_counted_per_split(tmp_path):
    """A complete test split must not make a short train split look complete."""
    ds = _store(tmp_path, shards=6250, max_size=1_500_000, split="train")
    processed = tmp_path / "processed"
    for idx in range(93_750):
        (processed / f"dimer_ap2_fused_test_spec_2_{idx}.pt").touch()
    assert len(ds._existing_shard_paths()) == 6250
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True


class _Stub:
    """Stands in for a stored Data object; only ``dimer_ind`` is consulted."""

    def __init__(self, dimer_ind):
        self.dimer_ind = dimer_ind


def _write_shard(ds, split, idx, dimer_inds):
    torch.save(
        [_Stub(i) for i in dimer_inds],
        osp.join(
            str(ds.root),
            "processed",
            f"dimer_ap2_fused_{split}_spec_2_{idx}{ds.file_extension}",
        ),
    )


def _aligned_store(tmp_path, *, shards, max_size, split="train"):
    ds = _store(tmp_path, shards=0, max_size=max_size, split=split)
    for idx in range(shards):
        _write_shard(
            ds,
            split,
            idx,
            range(idx * SHARD_OBJECTS, (idx + 1) * SHARD_OBJECTS),
        )
    return ds


def test_an_aligned_prefix_is_resumed_rather_than_rebuilt(tmp_path, capsys):
    ds = _aligned_store(tmp_path, shards=6250, max_size=1_500_000)
    ds.skip_processed = False
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True
    assert ds.skip_processed is True
    assert "skipped, not rebuilt" in capsys.readouterr().out


def test_a_prefix_that_dropped_a_dimer_forces_a_full_rebuild(tmp_path, capsys):
    """A dropped dimer desynchronises the raw index from the write index."""
    ds = _aligned_store(tmp_path, shards=10, max_size=1_500_000)
    _write_shard(ds, "train", 9, range(145, 145 + SHARD_OBJECTS))
    ds.skip_processed = False
    ds._request_extension_if_store_is_short()
    assert ds.force_reprocess is True
    assert ds.skip_processed is False
    assert "do not line up" in capsys.readouterr().out


def test_a_short_final_shard_forces_a_full_rebuild(tmp_path):
    ds = _aligned_store(tmp_path, shards=10, max_size=1_500_000)
    _write_shard(ds, "train", 9, range(144, 144 + 3))
    assert ds._prefix_is_index_aligned() is False


def test_objects_without_a_raw_index_force_a_full_rebuild(tmp_path):
    ds = _store(tmp_path, shards=4, max_size=1_500_000)
    assert ds._prefix_is_index_aligned() is False


def test_an_unreadable_final_shard_forces_a_full_rebuild(tmp_path):
    ds = _aligned_store(tmp_path, shards=4, max_size=1_500_000)
    with open(
        osp.join(str(tmp_path), "processed", "dimer_ap2_fused_train_spec_2_3.pt"), "wb"
    ) as fh:
        fh.write(b"not a torch file")
    with pytest.warns(UserWarning, match="lines up with the raw order"):
        assert ds._prefix_is_index_aligned() is False


def test_alignment_is_judged_from_the_last_shard_of_this_split(tmp_path):
    ds = _aligned_store(tmp_path, shards=6250, max_size=1_500_000, split="train")
    _write_shard(ds, "test", 0, range(9000, 9000 + SHARD_OBJECTS))
    assert ds._prefix_is_index_aligned() is True


def test_a_complete_store_is_never_asked_about_alignment(tmp_path):
    """Nothing is loaded when there is nothing to extend."""
    ds = _aligned_store(tmp_path, shards=6250, max_size=100_000)
    ds.skip_processed = False
    ds._request_extension_if_store_is_short()
    assert ds.skip_processed is False
    assert ds.force_reprocess is False
