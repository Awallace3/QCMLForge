import os
import os.path as osp
from threading import Lock


_LMDB_ENV_REGISTRY = {}
_LMDB_ENV_LOCK = Lock()


def _discard_if_inherited(env_path):
    """Return the registry entry for this process, forgetting a forked one.

    ``DataLoader`` workers are forked, so a child starts with a copy of this
    registry describing environments the parent opened.  Reusing such a handle
    keeps the child from opening its own, and every worker generation then
    consumes reader-table slots that are never reclaimed until the store hits
    ``MDB_READERS_FULL``.  Entries from another pid are dropped so the caller
    opens (or closes) its own environment instead.  Callers must hold
    ``_LMDB_ENV_LOCK``.
    """
    entry = _LMDB_ENV_REGISTRY.get(env_path)
    if entry is not None and entry["pid"] != os.getpid():
        _LMDB_ENV_REGISTRY.pop(env_path, None)
        return None
    return entry


def acquire_lmdb_env(
    lmdb_module,
    path,
    *,
    map_size,
    readonly,
    max_dbs=0,
    lock,
    max_readers=256,
):
    env_path = osp.abspath(path)

    with _LMDB_ENV_LOCK:
        entry = _discard_if_inherited(env_path)
        if entry is not None:
            if entry["readonly"] and not readonly:
                raise RuntimeError(
                    f"The environment '{env_path}' is already open read-only in this process."
                )

            entry["refcount"] += 1
            return entry["env"]

        env = lmdb_module.open(
            env_path,
            map_size=map_size,
            readonly=readonly,
            max_dbs=max_dbs,
            lock=lock,
            max_readers=max_readers,
        )
        _LMDB_ENV_REGISTRY[env_path] = {
            "env": env,
            "readonly": readonly,
            "refcount": 1,
            "pid": os.getpid(),
        }
        return env


def release_lmdb_env(path, env):
    if env is None:
        return

    env_path = osp.abspath(path)

    with _LMDB_ENV_LOCK:
        entry = _discard_if_inherited(env_path)
        if entry is None or entry["env"] is not env:
            env.close()
            return

        entry["refcount"] -= 1
        if entry["refcount"] > 0:
            return

        _LMDB_ENV_REGISTRY.pop(env_path, None)
        env.close()


def retain_lmdb_env(path, env):
    """Take an extra reference on an environment that is already open.

    Returns True when the registry entry was found and incremented.  An
    environment that this process did not acquire cannot be refcounted, so the
    caller is left with the pre-existing behaviour of sharing an unowned handle.
    """
    if env is None or path is None:
        return False

    env_path = osp.abspath(path)

    with _LMDB_ENV_LOCK:
        entry = _discard_if_inherited(env_path)
        if entry is None or entry["env"] is not env:
            return False

        entry["refcount"] += 1
        return True


class LmdbEnvHandleMixin:
    """Keep the LMDB environment alive across shallow copies of a dataset.

    ``torch_geometric.data.Dataset.index_select`` builds a subset with
    ``copy.copy(self)``.  Without this hook the subset shares ``lmdb_env`` while
    holding no registry reference, so dropping the original -- which
    ``ds = ds[:n]`` does -- runs ``__del__`` -> ``_close_lmdb`` ->
    ``release_lmdb_env``, the refcount reaches zero, and the environment closes
    underneath the live subset.  Cached attributes such as ``_length`` keep
    answering, so the failure only surfaces at the first uncached read.

    Mix in ahead of ``Dataset`` so this ``__copy__`` wins the MRO.
    """

    def __copy__(self):
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        retain_lmdb_env(
            getattr(self, "lmdb_path", None), getattr(self, "lmdb_env", None)
        )
        return clone
