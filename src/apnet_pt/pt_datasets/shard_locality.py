"""Shard-locality-aware sampling for the sharded on-disk dimer stores.

Why this exists
---------------
The store keeps ``datapoint_storage_n_objects`` dimers per ``.pt`` shard (16 on
the production CLIFF2 store) and ``Dataset.get`` deserialises a whole shard to
return one dimer, caching only the most recent one.  A sequential pass
therefore pays one read per 16 dimers; a uniformly shuffled pass pays one read
per dimer.  Measured on the production store (job 12379500), that is
1301.6 samples/s sequential against 79.4 shuffled -- a 16.4x read
amplification that is exactly the shard size.

This sampler recovers most of that without giving up shuffling.  It shuffles
*shards* rather than dimers, hands each loader worker its own disjoint block of
shards, shuffles the dimers within that block, and emits batches round-robin so
that block lands on the worker it was cut for.  A worker then reads each of its
``block_shards`` shards once and serves ``shard_size * block_shards`` dimers
out of its LRU -- read amplification 1x, with dimers mixed across a window of
``shard_size * block_shards`` rather than across the whole store.

The mixing window is the knob.  ``block_shards=256`` on a 16-dimer store mixes
over 4,096 dimers per worker per block, and which shards share a block is
re-drawn every epoch from a fresh permutation, so a dimer's batch-mates change
from epoch to epoch.  It is not the same distribution as a global shuffle, and
that is the trade this sampler exists to expose: it must be opted into.

Worker alignment
----------------
``DataLoader`` hands batch *b* to worker ``b % num_workers`` (round-robin over
``_index_queues``).  The interleaving below is built on that.  If it ever stops
holding, the sampler still emits every index exactly once -- the blocks just
stop lining up with the workers that cache them, and the win degrades toward
the unshuffled baseline.  Correctness does not depend on it.

Splitting across DDP ranks
--------------------------
Ranks are cut on *item* count, not shard count.  Striding the shard
permutation by rank is only an even split when every shard is equally
populated, and they are not: the final shard of a store is short, and an
index-selected dataset can leave any shard short.  Cutting the concatenated
item sequence instead splits at most one shard across a rank boundary -- so at
most ``num_replicas - 1`` shards are read twice in an epoch, against 93,750
shards on the production store -- and no rank has to discard a sample to match
the others' length.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler


def shard_ids_for_dataset(dataset, shard_size: int) -> np.ndarray:
    """Shard index of every sampler position in ``dataset``.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset the loader will read.  Only ``__len__`` and, when present,
        ``indices()`` are used.
    shard_size : int
        Dimers per shard (``datapoint_storage_n_objects``).

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length ``len(dataset)``; entry *i* is the shard
        holding the sample at position *i*.

    Notes
    -----
    A PyG dataset that has been sliced (``dataset[:max_size]``) or
    index-selected (the excluded-elements path) addresses its samples by
    position, while the shard a sample lives in is a property of its position
    in the *underlying* store.  ``indices()`` is that mapping; for an unsliced
    dataset it is the identity and this reduces to
    ``arange(len) // shard_size``.
    """
    try:
        underlying = np.asarray(dataset.indices(), dtype=np.int64)
    except Exception:
        underlying = np.arange(len(dataset), dtype=np.int64)
    if underlying.size != len(dataset):
        underlying = np.arange(len(dataset), dtype=np.int64)
    return underlying // int(shard_size)


def _wrapped_ranges(lo: int, count: int, n: int) -> List[Tuple[int, int]]:
    """Half-open ranges covering ``count`` positions from ``lo``, mod ``n``.

    Parameters
    ----------
    lo : int
        First global position to take.
    count : int
        How many positions to take.
    n : int
        Size of the space being wrapped over.

    Returns
    -------
    list of (int, int)
        Half-open ``[start, stop)`` ranges within ``[0, n)``.

    Notes
    -----
    Wrapping is how the last rank is padded, and it replaces repeating a
    rank's own indices back at it: ``num_samples * num_replicas`` exceeds the
    dataset by fewer than ``num_replicas`` items, and those few come from the
    front of the epoch's shard order rather than from whatever the short rank
    happened to hold.
    """
    if n <= 0 or count <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    pos = lo % n
    remaining = count
    while remaining > 0:
        take = min(remaining, n - pos)
        ranges.append((pos, pos + take))
        remaining -= take
        pos = 0
    return ranges


class ShardBlockSampler(Sampler):
    """Shuffle shards, not dimers.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset the loader will read.  Only ``__len__`` and, when present,
        ``indices()`` are used.
    shard_size : int
        Dimers per shard (``datapoint_storage_n_objects``).
    batch_size : int
        The loader's per-rank batch size.  Needed because the emission order
        has to be cut on batch boundaries for the round-robin worker
        assignment to line up.
    block_shards : int, default 256
        Shards per worker per block.  This is the mixing window, in shards,
        and should not exceed the dataset's shard-cache size -- a block larger
        than the cache is re-read rather than reused.
    num_workers : int, default 0
        The loader's ``num_workers`` (0 is treated as 1).
    seed : int, default 43
        Base seed; the permutation is a function of ``(seed, epoch)``.
    num_replicas : int, optional
        DDP world size.  ``None`` means single-process.
    rank : int, optional
        DDP rank.  ``None`` means single-process.
    drop_last : bool, default False
        Drop the ragged tail instead of emitting it unblocked.

    Notes
    -----
    Every rank draws exactly ``len(self)`` indices, which DDP requires: a
    short rank hangs the next epoch's collective.  Unlike
    ``DistributedSampler``, the ranks are cut on item count rather than on
    shard count, so meeting that requirement costs only the global remainder
    (fewer than ``num_replicas`` duplicated samples per epoch) instead of
    discarding whatever a heavier rank held.
    """

    def __init__(
        self,
        dataset,
        shard_size: int,
        batch_size: int,
        block_shards: int = 256,
        num_workers: int = 0,
        seed: int = 43,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.shard_size = max(1, int(shard_size))
        self.batch_size = max(1, int(batch_size))
        self.block_shards = max(1, int(block_shards))
        self.num_workers = max(1, int(num_workers))
        self.seed = int(seed)
        self.num_replicas = int(num_replicas) if num_replicas else 1
        self.rank = int(rank) if rank else 0
        self.drop_last = bool(drop_last)
        self.epoch = 0

        shard_ids = shard_ids_for_dataset(dataset, self.shard_size)
        self.n_items = int(shard_ids.size)
        # Positions grouped by shard, cheaply and once: a stable argsort puts
        # every shard's positions contiguous and in ascending order, and the
        # run boundaries fall out of the sorted key.
        order = np.argsort(shard_ids, kind="stable")
        keys = shard_ids[order]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        self._groups = np.split(order, bounds) if self.n_items else []
        self.n_shards = len(self._groups)
        # Shards are not equally populated -- the last one of a store is short,
        # and an index-selected dataset can leave any of them short -- so the
        # rank split below needs their sizes, not just their count.
        self._group_sizes = np.array(
            [g.size for g in self._groups], dtype=np.int64
        )

        if self.num_replicas > 1:
            # Every rank must draw the same number of indices or the next
            # epoch's collective deadlocks on the short rank.
            self.num_samples = -(-self.n_items // self.num_replicas)
        else:
            self.num_samples = self.n_items

    def set_epoch(self, epoch: int) -> None:
        """Re-draw the shard permutation for ``epoch``."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _rank_parts(self, shard_perm: np.ndarray) -> List[np.ndarray]:
        """This rank's share of ``shard_perm``, cut on item count.

        Parameters
        ----------
        shard_perm : numpy.ndarray
            This epoch's shard permutation.

        Returns
        -------
        list of numpy.ndarray
            Sampler positions, one array per shard the rank touches, in
            permutation order.  The arrays sum to exactly ``len(self)``.

        Notes
        -----
        A shard that straddles a rank boundary is sliced, so it is read by two
        ranks; there are at most ``num_replicas - 1`` such shards per epoch.
        That is the price of never discarding a sample, and it is cheap: two
        extra reads per epoch on a four-rank job.  With ``num_replicas == 1``
        the range is the whole dataset and every group comes back whole.
        """
        sizes = self._group_sizes[shard_perm]
        ends = np.cumsum(sizes)
        starts = ends - sizes
        parts: List[np.ndarray] = []
        for lo, hi in _wrapped_ranges(
            self.rank * self.num_samples, self.num_samples, self.n_items
        ):
            for k in np.flatnonzero((ends > lo) & (starts < hi)):
                group = self._groups[shard_perm[k]]
                begin, end = int(starts[k]), int(ends[k])
                # A boundary shard is sliced, not dropped; the slice is a
                # contiguous run of its positions, and the block shuffle
                # below mixes them anyway.
                a, b = max(lo, begin) - begin, min(hi, end) - begin
                parts.append(group[a:b])
        return parts

    def _epoch_indices(self) -> List[int]:
        """Sampler positions for the current epoch, in emission order."""
        if not self.n_shards:
            return []
        rng = np.random.default_rng([self.seed, self.epoch])
        shard_perm = rng.permutation(self.n_shards)
        parts: Sequence[np.ndarray] = self._rank_parts(shard_perm)

        nw = self.num_workers
        span = nw * self.block_shards
        # One carry buffer per worker: a block rarely divides evenly into
        # batches, and the remainder has to stay with its worker or the next
        # block's cache is cold for no reason.
        carry: List[List[int]] = [[] for _ in range(nw)]
        out: List[int] = []
        for start in range(0, len(parts), span):
            sup = parts[start:start + span]
            for w in range(nw):
                # Strided rather than contiguous so a short final super-block
                # still spreads its shards over all workers.
                chunk = sup[w::nw]
                items = (
                    np.concatenate(chunk) if chunk
                    else np.empty(0, dtype=np.int64)
                )
                rng.shuffle(items)
                carry[w].extend(items.tolist())
            # Emit only full rounds: a partial round would shift every later
            # batch onto a different worker and undo the alignment.
            while all(len(c) >= self.batch_size for c in carry):
                for w in range(nw):
                    out.extend(carry[w][: self.batch_size])
                    del carry[w][: self.batch_size]

        tail = [i for c in carry for i in c]
        if not self.drop_last and tail:
            rng.shuffle(tail)
            out.extend(tail)

        if self.num_replicas > 1 and len(out) < self.num_samples and out:
            # Only reachable under `drop_last`, which discards a ragged tail
            # whose length differs from rank to rank.  Without it the parts
            # already sum to `num_samples`, so there is nothing to pad and --
            # the point of cutting on item count -- nothing to truncate.
            need = self.num_samples - len(out)
            reps = -(-need // len(out))
            out = out + (out * reps)[:need]
        return out

    def __iter__(self) -> Iterator[int]:
        return iter(self._epoch_indices())
