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
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


def shard_ids_for_dataset(dataset, shard_size):
    """Shard index of every *sampler position* in ``dataset``.

    A PyG dataset that has been sliced (``dataset[:max_size]``) or index-selected
    (the excluded-elements path) addresses its samples by position, while the
    shard a sample lives in is a property of its position in the *underlying*
    store.  ``indices()`` is that mapping; for an unsliced dataset it is the
    identity and this reduces to ``arange(len) // shard_size``.
    """
    try:
        underlying = np.asarray(dataset.indices(), dtype=np.int64)
    except Exception:
        underlying = np.arange(len(dataset), dtype=np.int64)
    if underlying.size != len(dataset):
        underlying = np.arange(len(dataset), dtype=np.int64)
    return underlying // int(shard_size)


class ShardBlockSampler(Sampler):
    """Shuffle shards, not dimers.

    Args:
        dataset: the dataset the loader will read.  Only ``len`` and, when
            present, ``indices()`` are used.
        shard_size: dimers per shard (``datapoint_storage_n_objects``).
        batch_size: the loader's per-rank batch size.  Needed because the
            emission order has to be cut on batch boundaries for the
            round-robin worker assignment to line up.
        block_shards: shards per worker per block.  This is the mixing window,
            in shards, and should not exceed the dataset's shard-cache size --
            a block larger than the cache is re-read rather than reused.
        num_workers: the loader's ``num_workers`` (0 is treated as 1).
        seed: base seed; the permutation is a function of ``(seed, epoch)``.
        num_replicas / rank: set for DDP.  Shards are split across ranks before
            blocks are cut, so ranks never contend for the same shard.
        drop_last: drop the ragged tail instead of emitting it unblocked.
    """

    def __init__(
        self,
        dataset,
        shard_size,
        batch_size,
        block_shards=256,
        num_workers=0,
        seed=43,
        num_replicas=None,
        rank=None,
        drop_last=False,
    ):
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

        if self.num_replicas > 1:
            # Every rank must draw the same number of indices or the next
            # epoch's collective deadlocks on the short rank.
            self.num_samples = -(-self.n_items // self.num_replicas)
        else:
            self.num_samples = self.n_items

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def _epoch_indices(self):
        if not self.n_shards:
            return []
        rng = np.random.default_rng([self.seed, self.epoch])
        shard_perm = rng.permutation(self.n_shards)
        if self.num_replicas > 1:
            shard_perm = shard_perm[self.rank:: self.num_replicas]

        nw = self.num_workers
        span = nw * self.block_shards
        # One carry buffer per worker: a block rarely divides evenly into
        # batches, and the remainder has to stay with its worker or the next
        # block's cache is cold for no reason.
        carry = [[] for _ in range(nw)]
        out = []
        for start in range(0, len(shard_perm), span):
            sup = shard_perm[start:start + span]
            for w in range(nw):
                # Strided rather than contiguous so a short final super-block
                # still spreads its shards over all workers.
                items = np.concatenate(
                    [self._groups[s] for s in sup[w::nw]]
                ) if sup[w::nw].size else np.empty(0, dtype=np.int64)
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

        if self.num_replicas > 1:
            if len(out) < self.num_samples:
                # Pad the way DistributedSampler does: repeat from the front.
                need = self.num_samples - len(out)
                if out:
                    reps = -(-need // len(out))
                    out = out + (out * reps)[:need]
            out = out[: self.num_samples]
        return out

    def __iter__(self):
        return iter(self._epoch_indices())
