"""Shard-locality sampling and the multi-shard LRU.

Two claims are worth a test here.  The first is that the sampler is still a
sampler: every epoch is a permutation of the dataset, DDP ranks stay disjoint
and equal-length, and the ordering moves when the epoch does.  The second is
the reason it exists -- that with the LRU sized to the block, an epoch reads
each shard about once instead of once per dimer drawn from it.  That claim is
about I/O, so it is checked by counting reads through the real ``get``, with a
subclass standing in for the disk at the one seam the dataset exposes for it.

The default path is checked too, and is the strictest of these: at
``block_shards=0`` nothing is constructed and the shard cache is never
allocated, so an existing run's ordering is untouched.
"""

import os.path as osp
from collections import OrderedDict

import numpy as np
import pytest

from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_module_dataset
from apnet_pt.pt_datasets.shard_locality import (
    ShardBlockSampler,
    shard_ids_for_dataset,
)

SHARD_SIZE = 16
BATCH_SIZE = 128


class _FakeDataset:
    """Just enough dataset for the sampler: a length and an index map."""

    def __init__(self, n, indices=None):
        self.n = n
        self._indices = indices

    def __len__(self):
        return self.n

    def indices(self):
        return range(self.n) if self._indices is None else self._indices


def _sampler(n=40960, **kw):
    kw.setdefault("block_shards", 64)
    kw.setdefault("num_workers", 7)
    return ShardBlockSampler(
        _FakeDataset(n), shard_size=SHARD_SIZE, batch_size=BATCH_SIZE, **kw
    )


def _read_count(order, cache_size, num_workers, batch_size=BATCH_SIZE):
    """Shard reads an epoch in ``order`` would cost.

    Mirrors what the loader does: batch ``b`` goes to worker ``b % nw``, and
    each worker carries its own copy of the dataset and therefore its own LRU.
    """
    nw = max(1, num_workers)
    caches = [OrderedDict() for _ in range(nw)]
    reads = 0
    for start in range(0, len(order) - batch_size + 1, batch_size):
        cache = caches[(start // batch_size) % nw]
        for i in order[start:start + batch_size]:
            shard = i // SHARD_SIZE
            if shard in cache:
                cache.move_to_end(shard)
                continue
            reads += 1
            cache[shard] = True
            while len(cache) > cache_size:
                cache.popitem(last=False)
    return reads


# --- the sampler is still a sampler -----------------------------------------


@pytest.mark.parametrize("n", [40960, 40961, 1024])
@pytest.mark.parametrize("num_workers", [0, 4, 7])
def test_epoch_is_a_permutation(n, num_workers):
    s = _sampler(n, num_workers=num_workers)
    order = list(s)
    assert len(order) == n == len(s)
    assert sorted(order) == list(range(n))


def test_set_epoch_changes_the_order_and_is_reproducible():
    s = _sampler()
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    second = list(s)
    assert first != second
    s.set_epoch(0)
    assert list(s) == first


def test_ddp_ranks_are_disjoint_equal_and_cover_the_dataset():
    n, world = 40960, 4
    ranks = [
        _sampler(n, num_replicas=world, rank=r) for r in range(world)
    ]
    sets = [set(r) for r in ranks]
    assert len({len(r) for r in ranks}) == 1, "unequal lengths deadlock DDP"
    assert sum(len(s) for s in sets) == n
    assert set().union(*sets) == set(range(n))
    for a in range(world):
        for b in range(a + 1, world):
            assert not sets[a] & sets[b]


def _ddp_ranks(n, world, **kw):
    kw.setdefault("block_shards", 4)
    kw.setdefault("num_workers", 2)
    return [
        ShardBlockSampler(
            _FakeDataset(n), shard_size=SHARD_SIZE, batch_size=8,
            num_replicas=world, rank=r, **kw
        )
        for r in range(world)
    ]


@pytest.mark.parametrize(
    "n, world",
    [
        (100, 2),      # 6 full shards and a 4-dimer remainder
        (40961, 4),    # one item past a shard boundary
        (SHARD_SIZE, 2),   # fewer shards than ranks: one shard, two ranks
        (3, 4),        # fewer items than ranks
    ],
)
def test_every_sample_trains_every_epoch_under_ddp(n, world):
    """Ranks are cut on item count, so nothing is dropped to equalise them.

    Splitting the *shard* permutation by rank hands equal shard counts, which
    is equal item counts only on a store whose shards are all full.  When they
    are not, a heavier rank has to discard the overflow to match the others'
    length and those samples never train.  100 items in 16-dimer shards across
    two ranks used to lose 14 of them per epoch.
    """
    ranks = _ddp_ranks(n, world)
    emitted = [list(r) for r in ranks]

    assert len({len(e) for e in emitted}) == 1, "unequal lengths deadlock DDP"
    assert [len(e) for e in emitted] == [len(r) for r in ranks], (
        "__len__ must match what __iter__ actually yields"
    )
    assert set().union(*map(set, emitted)) == set(range(n)), "samples dropped"

    # Duplication is the price of equal lengths, and it is bounded by the
    # global remainder -- never by how unevenly the shards happened to fall.
    duplicated = sum(len(e) for e in emitted) - n
    assert duplicated == len(ranks[0]) * world - n
    assert duplicated < world


def test_ddp_shard_splitting_stays_rare():
    """At most one shard per rank boundary is read twice."""
    n, world = 40960, 4
    ranks = _ddp_ranks(n, world, block_shards=64, num_workers=4)
    owners = {}
    for r, sampler in enumerate(ranks):
        for i in sampler:
            owners.setdefault(i // SHARD_SIZE, set()).add(r)
    shared = [sh for sh, rs in owners.items() if len(rs) > 1]
    assert len(shared) <= world - 1, f"{len(shared)} shards split across ranks"


def test_subset_indices_are_mapped_to_underlying_shards():
    """A sliced dataset addresses samples by position, not by store index."""
    underlying = list(range(1000, 1000 + 320))
    ds = _FakeDataset(320, indices=underlying)
    ids = shard_ids_for_dataset(ds, SHARD_SIZE)
    assert ids[0] == 1000 // SHARD_SIZE
    assert ids.max() == (1000 + 319) // SHARD_SIZE
    s = ShardBlockSampler(ds, shard_size=SHARD_SIZE, batch_size=32,
                          block_shards=4, num_workers=2)
    assert sorted(s) == list(range(320))


# --- the reason it exists ----------------------------------------------------


@pytest.mark.parametrize("block_shards", [64, 256])
def test_read_amplification_collapses_to_about_one(block_shards):
    n = 163840
    s = _sampler(n, block_shards=block_shards, num_workers=7)
    order = list(s)
    n_shards = n // SHARD_SIZE

    blocked = _read_count(order, block_shards, 7) / n_shards
    shuffled = _read_count(
        np.random.default_rng(0).permutation(n).tolist(), block_shards, 7
    ) / n_shards

    assert blocked < 1.25, f"blocked amplification {blocked:.2f}"
    assert shuffled > 10, f"uniform shuffle amplification {shuffled:.2f}"
    assert shuffled / blocked > 8


def test_a_cache_smaller_than_the_block_gives_the_reads_back():
    """The two knobs are one knob; the loader ties them together for a reason."""
    order = list(_sampler(163840, block_shards=256, num_workers=7))
    n_shards = 163840 // SHARD_SIZE
    assert _read_count(order, 256, 7) / n_shards < 1.25
    assert _read_count(order, 1, 7) / n_shards > 10


def test_dimers_from_one_shard_do_not_all_land_in_one_batch():
    """Blocking is not batching: a shard's dimers still spread across batches."""
    order = list(_sampler(163840, block_shards=256, num_workers=7))
    batches = {}
    for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
        for i in order[start:start + BATCH_SIZE]:
            batches.setdefault(i // SHARD_SIZE, set()).add(start // BATCH_SIZE)
    spread = np.array([len(v) for v in batches.values()])
    assert spread.mean() > 10, f"mean spread {spread.mean():.2f} of {SHARD_SIZE}"


def test_batch_mates_are_redrawn_every_epoch():
    s = _sampler(163840, block_shards=256, num_workers=7)

    def mates(epoch):
        s.set_epoch(epoch)
        order = list(s)
        out = {}
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            block = order[start:start + BATCH_SIZE]
            as_set = set(block)
            for i in block:
                out[i] = as_set
        return out

    a, b = mates(0), mates(1)
    keys = [k for k in list(a)[:2000] if k in b]
    overlap = np.mean([len(a[k] & b[k]) for k in keys])
    assert overlap < 0.05 * BATCH_SIZE, f"mean batch-mate overlap {overlap:.2f}"


# --- the multi-shard LRU, through the real get() -----------------------------


class _CountingStore(ap2_fused_module_dataset):
    """An ``ap2_fused_module_dataset`` whose shards are counters, not files.

    Overrides the one seam the dataset exposes for the read, so ``get`` and
    the LRU under test are the production ones and nothing outside this class
    is altered.  Replacing ``torch.load`` module-wide would have reached far
    past the read being counted, and the repo rules that style out.
    """

    PREFIX = "dimer_ap2_fused_spec_2_"

    def __init__(self, root, n_shards, cache_size):
        # Deliberately not calling ``super().__init__``: that processes a
        # dataset onto disk, and the point here is that nothing is on disk.
        self.root = str(root)
        self.datapoint_storage_n_objects = SHARD_SIZE
        self.spec_type = 2
        self.split_db = False
        self.split = "all"
        self.storage_type = "pt"
        self.active_idx_data = None
        self.active_data = None
        self.shard_cache_size = 1
        self._shard_cache = None
        self._n_shards = n_shards
        self.reads = []
        self.set_shard_cache_size(cache_size)
        assert self.split_name == "" and self.file_extension == ".pt"

    def _load_shard(self, datapath):
        name = osp.basename(datapath)
        assert name.startswith(self.PREFIX) and name.endswith(".pt"), name
        shard = int(name[len(self.PREFIX):-3])
        assert 0 <= shard < self._n_shards
        self.reads.append(shard)
        return [(shard, j) for j in range(SHARD_SIZE)]


def _stub_store(tmp_path, n_shards, cache_size):
    ds = _CountingStore(tmp_path, n_shards, cache_size)
    return ds, ds.reads


def test_lru_is_absent_by_default_and_get_is_unchanged(tmp_path):
    ds, reads = _stub_store(tmp_path, 8, cache_size=1)
    assert ds._shard_cache is None
    # Alternating between two shards with no cache re-reads on every switch --
    # this is the behaviour the default must keep.
    for i in [0, 16, 1, 17, 2, 18]:
        ds.get(i)
    assert reads == [0, 1, 0, 1, 0, 1]


def test_lru_serves_repeat_shards_and_respects_capacity(tmp_path):
    ds, reads = _stub_store(tmp_path, 8, cache_size=4)
    for i in [0, 16, 1, 17, 2, 18]:
        assert ds.get(i) == (i // SHARD_SIZE, i % SHARD_SIZE)
    assert reads == [0, 1], "a cached shard must not be re-read"
    assert len(ds._shard_cache) == 2

    # Walk past the capacity; the oldest shard is the one that goes.
    for shard in range(6):
        ds.get(shard * SHARD_SIZE)
    assert len(ds._shard_cache) == 4
    assert list(ds._shard_cache) == [2, 3, 4, 5]


def test_set_shard_cache_size_shrinks_in_place(tmp_path):
    ds, _ = _stub_store(tmp_path, 8, cache_size=4)
    for shard in range(4):
        ds.get(shard * SHARD_SIZE)
    assert len(ds._shard_cache) == 4
    ds.set_shard_cache_size(2)
    assert list(ds._shard_cache) == [2, 3]
    ds.set_shard_cache_size(1)
    assert ds._shard_cache is None
