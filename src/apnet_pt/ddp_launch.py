"""Rendezvous resolution for externally launched (``srun``/``torchrun``) DDP.

One place decides *how a rank learns who it is*, so a single-node job and a
multi-node job take the identical code path.  ``srun`` gives every task
``SLURM_PROCID``/``SLURM_LOCALID``/``SLURM_NTASKS`` but no rendezvous endpoint,
so the endpoint has to be derived: the first hostname of the allocation is the
only value every task can compute without communicating, and the job id is the
only per-job unique number available for the port.

Resolution order is explicit argument, then environment, then SLURM, then a
single-host default.  That order is what makes the module usable off-cluster
(nothing SLURM-specific is required) while still being correct for a two-node
allocation, where ``MASTER_ADDR=localhost`` would leave rank 0 talking to
itself and every other node waiting forever.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

# torch's own default when nothing else is known.
DEFAULT_MASTER_PORT = 29500
# Ephemeral-range-avoiding window for job-id-derived ports.  20000-39999 sits
# above the privileged range and below the usual ``ip_local_port_range`` floor
# (32768 on Linux is the common default, so the top of this window can overlap;
# a collision only costs a bind retry at the next job id).
_DERIVED_PORT_BASE = 20000
_DERIVED_PORT_SPAN = 20000


@dataclass(frozen=True)
class Rendezvous:
    """Everything a rank needs to join a process group."""

    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: int
    # How the endpoint was decided, purely so logs can say so.
    source: str = "default"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _int_env(*names: str) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def slurm_first_hostname() -> str | None:
    """First hostname of ``SLURM_JOB_NODELIST``, or ``None``.

    ``scontrol`` is the only reliable expander of the compressed nodelist
    syntax (``atl1-1-02-004-[1-2]``), so it is called rather than parsed.  Every
    failure mode -- no allocation, no ``scontrol`` on PATH, a non-zero exit --
    returns ``None`` so the caller falls back instead of dying.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get(
        "SLURM_NODELIST"
    )
    if not nodelist:
        return None
    if shutil.which("scontrol") is None:
        # Uncompressed single-node lists are the common case where scontrol is
        # missing (a login-node smoke test), and splitting on ',' is exact for
        # them.  A compressed list would be wrong, so it is refused.
        if "[" in nodelist:
            return None
        first = nodelist.split(",")[0].strip()
        return first or None
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        host = line.strip()
        if host:
            return host
    return None


def resolve_master_addr(explicit: str | None = None) -> tuple[str, str]:
    """``(address, source)`` for the rendezvous host."""
    if explicit:
        return explicit, "argument"
    env = os.environ.get("MASTER_ADDR")
    if env:
        return env, "env:MASTER_ADDR"
    host = slurm_first_hostname()
    if host:
        return host, "slurm:SLURM_JOB_NODELIST"
    return "localhost", "default"


def resolve_master_port(explicit: int | str | None = None) -> tuple[int, str]:
    """``(port, source)`` for the rendezvous port.

    A job-id-derived port keeps two of the user's jobs that land on the same
    node from fighting over one socket, which is a real failure mode on a
    shared cluster and shows up as a rendezvous timeout rather than as
    anything that names a port.
    """
    if explicit not in (None, ""):
        return int(explicit), "argument"
    env = os.environ.get("MASTER_PORT")
    if env:
        return int(env), "env:MASTER_PORT"
    job_id = _int_env("SLURM_JOB_ID", "SLURM_JOBID")
    if job_id is not None:
        return (
            _DERIVED_PORT_BASE + (job_id % _DERIVED_PORT_SPAN),
            "slurm:SLURM_JOB_ID",
        )
    return DEFAULT_MASTER_PORT, "default"


def resolve_rendezvous(
    *,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    master_addr: str | None = None,
    master_port: int | str | None = None,
) -> Rendezvous:
    """Fill in whatever the launcher did not pass explicitly.

    ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` are read before their ``SLURM_*``
    equivalents so a ``torchrun`` launch keeps working unchanged; ``srun``
    provides only the latter.
    """
    if rank is None:
        rank = _int_env("RANK", "SLURM_PROCID") or 0
    if local_rank is None:
        local_rank = _int_env("LOCAL_RANK", "SLURM_LOCALID") or 0
    if world_size is None:
        world_size = _int_env("WORLD_SIZE", "SLURM_NTASKS") or 1
    addr, addr_source = resolve_master_addr(master_addr)
    port, port_source = resolve_master_port(master_port)
    return Rendezvous(
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size),
        master_addr=addr,
        master_port=port,
        source=f"addr={addr_source} port={port_source}",
    )


def export_rendezvous(
    rendezvous: Rendezvous, *, omp_num_threads: int | None = None
) -> Rendezvous:
    """Publish ``rendezvous`` into the environment for ``env://`` init.

    ``init_process_group`` reads the environment, not arguments, so the
    resolution above only takes effect once it is exported.  Doing it here
    rather than at each call site means a rank cannot half-configure itself.
    """
    os.environ["RANK"] = str(rendezvous.rank)
    os.environ["LOCAL_RANK"] = str(rendezvous.local_rank)
    os.environ["WORLD_SIZE"] = str(rendezvous.world_size)
    os.environ["MASTER_ADDR"] = rendezvous.master_addr
    os.environ["MASTER_PORT"] = str(rendezvous.master_port)
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
    return rendezvous


def describe_rendezvous(rendezvous: Rendezvous) -> str:
    """One-line, greppable summary for a job log."""
    return (
        f"DDP rendezvous: rank={rendezvous.rank} "
        f"local_rank={rendezvous.local_rank} "
        f"world_size={rendezvous.world_size} "
        f"master={rendezvous.master_addr}:{rendezvous.master_port} "
        f"({rendezvous.source})"
    )
