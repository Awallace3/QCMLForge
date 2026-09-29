"""Shallow copies of LMDB-backed datasets must not close the environment.

``ds = ds[:n]`` builds the subset with ``copy.copy(self)`` and then drops the
original; before ``LmdbEnvHandleMixin`` the subset shared ``lmdb_env`` without a
registry reference, so the original's ``__del__`` closed the environment under
it and the next uncached read raised ``lmdb.Error: Attempt to operate on
closed/deleted/dropped object.``
"""

import copy
import gc
import os
import sys

import lmdb
import pytest
from torch_geometric.data import Data, Dataset

from apnet_pt import lmdb_utils
from apnet_pt.lmdb_utils import (
    LmdbEnvHandleMixin,
    acquire_lmdb_env,
    release_lmdb_env,
    retain_lmdb_env,
)


def _write_store(path, n_rows):
    env = lmdb.open(str(path), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        for i in range(n_rows):
            txn.put(str(i).encode(), str(i).encode())
    env.close()


class _FakeLmdbDataset(LmdbEnvHandleMixin, Dataset):
    """The production lifecycle in miniature: acquire, cached len, lazy get."""

    def __init__(self, lmdb_path, n_rows):
        self.lmdb_path = str(lmdb_path)
        self._length = n_rows
        self.lmdb_env = acquire_lmdb_env(
            lmdb,
            self.lmdb_path,
            map_size=1 << 24,
            readonly=False,
            lock=True,
        )
        super().__init__(root=None)

    def len(self):
        # Cached, exactly like the production classes -- a closed environment
        # still answers this, which is why the bug hid until the first read.
        return self._length

    def get(self, idx):
        with self.lmdb_env.begin() as txn:
            payload = txn.get(str(idx).encode())
        return Data(value=int(payload.decode()))

    def _close_lmdb(self):
        if self.lmdb_env is not None:
            release_lmdb_env(self.lmdb_path, self.lmdb_env)
            self.lmdb_env = None

    def __del__(self):
        try:
            self._close_lmdb()
        except Exception:
            pass


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "store"
    _write_store(path, 8)
    yield path
    for entry in list(lmdb_utils._LMDB_ENV_REGISTRY):
        lmdb_utils._LMDB_ENV_REGISTRY.pop(entry)["env"].close()


def _refcount(path):
    import os.path as osp

    entry = lmdb_utils._LMDB_ENV_REGISTRY.get(osp.abspath(str(path)))
    return None if entry is None else entry["refcount"]


def test_slice_survives_dropping_the_original(store):
    ds = _FakeLmdbDataset(store, 8)
    assert _refcount(store) == 1

    ds = ds[:4]  # torch_geometric index_select -> copy.copy, original dropped
    gc.collect()

    assert _refcount(store) == 1
    assert len(ds) == 4
    assert ds[3].value == 3  # uncached read through the surviving handle


def test_copy_takes_a_reference_and_release_is_balanced(store):
    ds = _FakeLmdbDataset(store, 8)
    clone = copy.copy(ds)
    assert _refcount(store) == 2

    del clone
    gc.collect()
    assert _refcount(store) == 1
    assert ds[1].value == 1

    del ds
    gc.collect()
    assert _refcount(store) is None


def test_retain_declines_environments_it_does_not_own(store, tmp_path):
    env = lmdb.open(str(store), map_size=1 << 24, subdir=True)
    try:
        assert retain_lmdb_env(str(store), env) is False
        assert retain_lmdb_env(str(store), None) is False
        assert retain_lmdb_env(None, env) is False
        assert _refcount(store) is None
    finally:
        env.close()

    owned = acquire_lmdb_env(
        lmdb, str(store), map_size=1 << 24, readonly=False, lock=True
    )
    try:
        assert retain_lmdb_env(str(store), owned) is True
        assert _refcount(store) == 2
        # A foreign env at a registered path must not touch the refcount.
        assert retain_lmdb_env(str(store), object()) is False
        assert _refcount(store) == 2
    finally:
        release_lmdb_env(str(store), owned)
        release_lmdb_env(str(store), owned)


def test_production_lmdb_datasets_use_the_copy_safe_hook():
    from apnet_pt.atomic_datasets import (
        atomic_hirshfeld_valencewdith_only_module_dataset,
        atomic_module_dataset_lmdb,
    )
    from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_module_dataset_lmdb
    from apnet_pt.pt_datasets.ap3_fused_ds import ap3_fused_module_dataset_lmdb
    from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
        ap3_fused_fsapt_module_dataset_lmdb,
    )

    for cls in (
        ap3_fused_module_dataset_lmdb,
        ap2_fused_module_dataset_lmdb,
        ap3_fused_fsapt_module_dataset_lmdb,
        atomic_module_dataset_lmdb,
        atomic_hirshfeld_valencewdith_only_module_dataset,
    ):
        assert issubclass(cls, LmdbEnvHandleMixin), cls.__name__
        # The mixin must win the MRO; several of these also define
        # __getstate__, which copy.copy would otherwise use to close the env.
        assert cls.__copy__ is LmdbEnvHandleMixin.__copy__, cls.__name__


def test_forked_child_opens_its_own_environment(store):
    """A DataLoader worker must not reuse the handle it inherited across fork.

    Reusing it keeps every worker generation on the parent's reader-table slots,
    which are never reclaimed, and the store eventually raises
    ``MDB_READERS_FULL`` mid-epoch.
    """
    parent_env = acquire_lmdb_env(
        lmdb, str(store), map_size=1 << 24, readonly=False, lock=True
    )
    try:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            os.close(read_fd)
            try:
                # The inherited entry must not be retainable ...
                retained = retain_lmdb_env(str(store), parent_env)
                # ... and dropping the inherited handle, as a worker does in
                # _check_worker_init, must leave nothing behind for the
                # following acquire to reuse.
                release_lmdb_env(str(store), parent_env)
                child_env = acquire_lmdb_env(
                    lmdb, str(store), map_size=1 << 24, readonly=False, lock=True
                )
                verdict = (
                    b"1"
                    if (child_env is not parent_env and not retained)
                    else b"0"
                )
                os.write(write_fd, verdict)
            except BaseException as exc:
                os.write(write_fd, b"e")
                sys.stderr.write(f"child failed: {exc!r}\n")
            finally:
                os._exit(0)

        os.close(write_fd)
        verdict = os.read(read_fd, 1)
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert verdict == b"1"
        # The child's bookkeeping is private to its address space.
        assert _refcount(store) == 1
    finally:
        release_lmdb_env(str(store), parent_env)
