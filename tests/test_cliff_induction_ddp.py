"""Multi-process training for the CLIFF parameter heads.

``AM_DimerParam_Model.train`` used to raise ``NotImplementedError`` above one
process, so every one of these behaviours is new and none of it is covered
anywhere else.  Two kinds of test live here.

The first kind actually runs it: ``mp.spawn`` two gloo/CPU ranks over a
four-dimer synthetic dataset and check what came out.  That is worth the seconds
it costs because the failure mode this work is most exposed to is silent.  The
epoch loop's forward is ``self.dimer_model(batch)``, not ``self.model(batch)``,
so a ``DDP(self.model)`` that is not also rebound into ``dimer_model`` fires no
hook, synchronizes no gradient, and produces a perfectly plausible loss curve
from N independent single-process runs.  Nothing short of comparing parameters
across ranks catches it, so ``test_ddp_ranks_stay_bitwise_identical`` does.

The second kind reads the source.  Rank-divergent control flow (an early
``break`` on NaN, a rank-0-only write) either hangs the job or corrupts a
checkpoint, and both are far cheaper to assert about statically than to
reproduce.  This mirrors ``test_cliff_induction_golden.py``, which guards the
resume sidecar the same way.
"""

import inspect
import os
import pickle

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from apnet_pt import ddp_launch, model_io
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels import mtp_mtp
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    AM_DimerParam_Model,
    AtomTypeParamNN,
    CliffClassicalOverlapModel,
)
from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_collate_update

from .conftest import _make_collate_item


# ---------------------------------------------------------------------------
# Rendezvous resolution
# ---------------------------------------------------------------------------


ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
)


@pytest.fixture
def clean_ddp_env(monkeypatch):
    """A rendezvous environment with nothing in it, restored afterwards.

    ``monkeypatch.delenv`` alone is not enough, and the difference cost two
    unrelated failures elsewhere in the suite. ``export_rendezvous`` writes to
    ``os.environ`` directly, and monkeypatch can only undo keys it was told
    about -- deleting a key that was never set records nothing to restore, so
    the value the test then publishes leaks into every later test in the
    session. ``tests/test_pol_mp.py`` and ``tests/test_precomputed_induced_
    dipole.py`` both call ``init_process_group`` and both inherited
    ``MASTER_PORT=7777`` from here, whereupon the second one died with
    ``EADDRINUSE``. Snapshot and restore explicitly.
    """
    watched = ENV_KEYS + ("OMP_NUM_THREADS",)
    saved = {key: os.environ.get(key) for key in watched}
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    try:
        yield monkeypatch
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_rendezvous_defaults_to_single_process(clean_ddp_env):
    rv = ddp_launch.resolve_rendezvous()
    assert (rv.rank, rv.local_rank, rv.world_size) == (0, 0, 1)
    assert rv.is_distributed is False
    assert rv.master_addr == "localhost"
    assert rv.master_port == 29500


def test_rendezvous_reads_slurm_when_torchrun_vars_are_absent(clean_ddp_env):
    """The multi-node case: no torchrun, only SLURM's own variables.

    ``MASTER_ADDR`` is the first host of the allocation rather than
    ``localhost``, which is the whole point -- a rank on the second node that
    rendezvouses with ``localhost`` waits for a peer that is not there, and the
    job hangs until the walltime rather than failing.
    """
    clean_ddp_env.setenv("SLURM_PROCID", "3")
    clean_ddp_env.setenv("SLURM_LOCALID", "1")
    clean_ddp_env.setenv("SLURM_NTASKS", "4")
    clean_ddp_env.setenv("SLURM_JOB_ID", "12345678")
    clean_ddp_env.setenv("SLURM_JOB_NODELIST", "atl1-1-02-007-1-0")

    rv = ddp_launch.resolve_rendezvous()
    assert (rv.rank, rv.local_rank, rv.world_size) == (3, 1, 4)
    assert rv.is_distributed is True
    assert rv.master_addr == "atl1-1-02-007-1-0"
    # Job-derived rather than a fixed 29500, so two of the user's jobs sharing
    # a node do not fight over one port.
    assert rv.master_port == 20000 + 12345678 % 20000


def test_rendezvous_prefers_explicit_arguments(clean_ddp_env):
    clean_ddp_env.setenv("RANK", "1")
    clean_ddp_env.setenv("WORLD_SIZE", "2")
    rv = ddp_launch.resolve_rendezvous(
        rank=0, local_rank=0, world_size=1, master_addr="host-a", master_port=1234
    )
    assert (rv.rank, rv.world_size, rv.master_addr, rv.master_port) == (
        0,
        1,
        "host-a",
        1234,
    )


def test_export_rendezvous_publishes_env_for_downstream_init(clean_ddp_env):
    """Exported so a nested ``init_process_group("nccl", init_method="env://")``
    sees the same endpoint the launcher chose."""
    rv = ddp_launch.export_rendezvous(
        ddp_launch.resolve_rendezvous(
            rank=1, local_rank=1, world_size=2, master_addr="h", master_port=7777
        ),
        omp_num_threads=3,
    )
    assert os.environ["RANK"] == "1"
    assert os.environ["LOCAL_RANK"] == "1"
    assert os.environ["WORLD_SIZE"] == "2"
    assert os.environ["MASTER_ADDR"] == "h"
    assert os.environ["MASTER_PORT"] == "7777"
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert rv.rank == 1


def test_train_ddp_slurm_and_train_models_share_one_resolver():
    """Two entry points, one rendezvous. Divergence here is a multi-node hang."""
    import train_ddp_slurm

    assert "ddp_launch.resolve_rendezvous" in inspect.getsource(
        train_ddp_slurm.setup_distributed
    )
    train_models_source = open(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_models.py")
    ).read()
    assert "ddp_launch.resolve_rendezvous()" in train_models_source


# ---------------------------------------------------------------------------
# A real two-rank run
# ---------------------------------------------------------------------------


def _build_harness(save_path=None):
    """A CLIFF overlap harness with no dataset, no GPU and tiny layers.

    Constructed inside each spawned worker rather than passed to it: the harness
    holds compiled submodules and an on-disk dataset handle in the real case,
    and pickling it into the child is exactly the fragility this avoids.
    """
    nested = AtomTypeParamNN(
        atom_model=AtomMPNN(
            n_message=1, n_rbf=2, n_neuron=8, n_embed=4, r_cut=5.0
        ),
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1.0, 0.4],
        param_start_std=[0.0, 0.0],
        n_params=2,
        freeze_atom_model=False,
    )
    for parameter in nested.parameters():
        torch.nn.init.constant_(parameter, 0.01)
    harness = CliffClassicalOverlapModel(
        atom_model=nested,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        ds_root=None if save_path is None else str(save_path),
    )
    return harness


def _tiny_split():
    """Four train dimers and two validation dimers, distinct targets.

    Four is the smallest count that divides evenly by two ranks *and* gives each
    rank more than one batch, so an epoch really performs several all-reduces.
    """
    train = [_make_collate_item(scale) for scale in (1.0, 1.3, 0.7, 1.6)]
    test = [_make_collate_item(scale) for scale in (1.1, 0.9)]
    return train, test


def _ddp_worker(rank, world_size, port, tmpdir, n_epochs, result_queue):
    """One rank of the real two-rank run, launched by ``mp.spawn``."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, init_method="env://"
        )
        torch.manual_seed(43)
        harness = _build_harness()
        train, test = _tiny_split()
        # The pre-training warmup forward runs the *parameter head*, which takes
        # a monomer-side atomic batch, not an un-collated dimer item. Stubbed
        # here for the same reason the exchange harness tests stub it: the real
        # `example_input` reads the on-disk dataset this harness does not have.
        warmup_batch = ap2_fused_collate_update(
            [_make_collate_item(1.0)]
        ).batch_atomic_A
        harness.example_input = lambda: warmup_batch
        harness.compile_model = lambda: None
        harness.model_save_path = os.path.join(tmpdir, "cliff_ddp.pt")
        # Derived, not passed: the sidecar path is a function of the checkpoint
        # path, which is what makes the chunk chain able to find it.
        train_state_file = model_io.train_state_path(harness.model_save_path)

        set_epochs = []
        original_sampler_cls = mtp_mtp.DistributedSampler

        class RecordingSampler(original_sampler_cls):
            def set_epoch(self, epoch):
                set_epochs.append(epoch)
                super().set_epoch(epoch)

        mtp_mtp.DistributedSampler = RecordingSampler
        try:
            harness.single_proc_train(
                train_dataset=train,
                test_dataset=test,
                n_epochs=n_epochs,
                batch_size=1,
                lr=1e-3,
                pin_memory=False,
                num_workers=0,
                skip_compile=True,
                rank=rank,
                world_size=world_size,
            )
        finally:
            mtp_mtp.DistributedSampler = original_sampler_cls

        head = model_io.unwrap_model(harness.model)
        flat = torch.cat(
            [p.detach().reshape(-1).double() for p in head.parameters()]
        )
        result_queue.put(
            {
                "rank": rank,
                "checksum": float(flat.sum()),
                "n_parameters": int(flat.numel()),
                "set_epochs": set_epochs,
                "checkpoint_exists": os.path.exists(harness.model_save_path),
                "sidecar_exists": os.path.exists(train_state_file),
                "model_is_ddp_after_train": isinstance(
                    harness.model, torch.nn.parallel.DistributedDataParallel
                ),
                "backend": dist.get_backend(),
                "world_size_seen": dist.get_world_size(),
            }
        )
        dist.barrier()
    except BaseException as exc:  # reported, not swallowed
        import traceback

        result_queue.put({"rank": rank, "error": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_ranks(tmpdir, world_size=2, n_epochs=2):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    # A fixed high port keyed off the pid: the CI box may be running another
    # copy of this suite.
    port = 30000 + os.getpid() % 20000
    mp.spawn(
        _ddp_worker,
        args=(world_size, port, str(tmpdir), n_epochs, queue),
        nprocs=world_size,
        join=True,
    )
    results = {}
    while len(results) < world_size:
        payload = queue.get(timeout=60)
        if "error" in payload:
            pytest.fail(f"rank {payload['rank']} raised:\n{payload['error']}")
        results[payload["rank"]] = payload
    return results


@pytest.fixture(scope="module")
def two_rank_run(tmp_path_factory):
    """One real two-rank run, shared by the assertions below.

    Module-scoped because spawning two interpreters that import torch costs
    several seconds and every assertion here reads the same run.
    """
    tmpdir = tmp_path_factory.mktemp("cliff_ddp")
    return _run_two_ranks(tmpdir, n_epochs=2), tmpdir


@pytest.fixture(scope="module")
def two_rank_resumed_run(two_rank_run):
    """A second two-rank job over the first one's output directory.

    This is the chunk boundary, which is the only place DDP and the resume
    sidecar interact: both ranks read the same sidecar written by rank 0, so
    their weights, their Adam moments and their epoch counter must agree, and
    the sampler must continue the global epoch sequence rather than replaying
    epoch 0's shuffle.
    """
    _, tmpdir = two_rank_run
    return _run_two_ranks(tmpdir, n_epochs=1), tmpdir


def test_ddp_two_ranks_complete_a_run(two_rank_run):
    results, _ = two_rank_run
    assert set(results) == {0, 1}
    for rank, payload in results.items():
        assert payload["backend"] == "gloo"
        assert payload["world_size_seen"] == 2
        assert payload["n_parameters"] > 0


def test_ddp_ranks_stay_bitwise_identical(two_rank_run):
    """The test the whole exercise exists for.

    Identical parameters after two epochs of *different* shards means the
    gradients were really all-reduced. If the DDP wrapper were not rebound into
    ``dimer_model.AtomTypeParam``, each rank would have descended on its own two
    dimers and these checksums would differ -- while both logs looked fine.
    """
    results, _ = two_rank_run
    assert results[0]["n_parameters"] == results[1]["n_parameters"]
    assert results[0]["checksum"] == results[1]["checksum"]


def test_ddp_sets_the_sampler_epoch_every_epoch(two_rank_run):
    """Without this the shuffle repeats, and a resumed chunk repeats it again."""
    results, _ = two_rank_run
    for payload in results.values():
        assert payload["set_epochs"] == [0, 1]


def test_ddp_writes_one_checkpoint_and_one_sidecar_from_rank_zero(two_rank_run):
    results, _ = two_rank_run
    assert results[0]["checkpoint_exists"] is True
    assert results[0]["sidecar_exists"] is True
    # Rank 1 sees the same paths on a shared filesystem, so existence proves
    # nothing about the writer; that is what the source contract below is for.
    assert results[1]["checkpoint_exists"] is True


def test_ddp_leaves_the_harness_unwrapped(two_rank_run):
    """A run ends in the shape a single-process run leaves it in.

    Otherwise the caller's ``harness.model.get_config()`` -- which
    ``_create_checkpoint`` and the predict paths both use -- fails on the DDP
    wrapper, which does not proxy attributes.
    """
    results, _ = two_rank_run
    for payload in results.values():
        assert payload["model_is_ddp_after_train"] is False


def test_ddp_checkpoint_loads_back(two_rank_run):
    results, tmpdir = two_rank_run
    assert results  # the run happened
    checkpoint = torch.load(
        os.path.join(str(tmpdir), "cliff_ddp.pt"),
        map_location="cpu",
        weights_only=False,
    )
    # No "module." prefixes: the checkpoint is of the unwrapped head, so it is
    # loadable by a single-process predict run.
    state = checkpoint["model_state_dict"]
    assert not any(key.startswith("module.") for key in state)


def test_ddp_checkpoint_keeps_the_atom_model_aliased(two_rank_run):
    """The checkpoint must not get bigger just because DDP wrote it.

    ``self.atom_model`` *is* ``self.model.atom_model``, so the embedded
    submodel state_dict shares storages with the top-level one and
    ``torch.save`` writes each storage once. Copying the two modules with two
    separate ``deepcopy`` calls breaks that sharing, and the artifact silently
    grows from 7.3 MB to 13.7 MB with byte-identical contents -- a deliverable
    whose size depends on the launch topology. Storage identity survives
    ``torch.save``/``torch.load``, so the loaded checkpoint still shows it.
    """
    results, tmpdir = two_rank_run
    assert results  # the run happened
    checkpoint = torch.load(
        os.path.join(str(tmpdir), "cliff_ddp.pt"),
        map_location="cpu",
        weights_only=False,
    )
    top = checkpoint["model_state_dict"]
    embedded = checkpoint["submodels"]["atom_model"]["model_state_dict"]
    shared = 0
    for key, tensor in embedded.items():
        outer = top.get(f"atom_model.{key}")
        if outer is None:
            continue
        assert outer.untyped_storage().data_ptr() == (
            tensor.untyped_storage().data_ptr()
        ), f"{key} was duplicated instead of shared"
        shared += 1
    assert shared > 0, "no embedded atom-model tensor matched a top-level one"


def test_ddp_sidecar_records_the_global_epoch(two_rank_run):
    """Resume is the reason the chunk chain works; DDP must not change it."""
    results, tmpdir = two_rank_run
    assert results
    payload = torch.load(
        model_io.train_state_path(os.path.join(str(tmpdir), "cliff_ddp.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["train_state_version"] == model_io.TRAIN_STATE_VERSION
    assert payload["epochs_completed"] == 2
    # Unwrapped, so a single-process chunk can resume from a DDP chunk's sidecar
    # and vice versa. `save_train_state` unwraps for us; this proves the DDP
    # wrapper did not slip a "module." prefix past it.
    assert not any(
        key.startswith("module.") for key in payload["model_state_dict"]
    )
    assert payload["identity"]["dimer_eval_type"] == "cliff_classical_overlap"
    # Historical defaults stay absent so old sidecars remain compatible.
    assert "induction_convergence_threshold" not in payload["identity"]
    assert "induction_max_iterations" not in payload["identity"]


def test_nondefault_scf_controls_are_resume_identity():
    source = inspect.getsource(AM_DimerParam_Model.single_proc_train)
    assert '"induction_convergence_threshold": solver_threshold' in source
    assert '"induction_max_iterations": solver_max_iterations' in source
    assert "DEFAULT_INDUCTION_CONVERGENCE_THRESHOLD" in source
    assert "DEFAULT_INDUCTION_MAX_ITERATIONS" in source


def test_ddp_resume_continues_the_epoch_sequence(two_rank_resumed_run):
    """The chunk boundary under DDP.

    ``set_epoch(2)`` rather than ``set_epoch(0)``: a chunk that restarts the
    sampler's epoch counter replays the first chunk's shuffle, so a long chain
    sees one epoch's ordering over and over and nothing in the log says so.
    """
    results, _ = two_rank_resumed_run
    for payload in results.values():
        assert payload["set_epochs"] == [2]


def test_ddp_resumed_ranks_stay_bitwise_identical(two_rank_resumed_run):
    """Both ranks loaded the same sidecar and stayed in step through it."""
    results, _ = two_rank_resumed_run
    assert results[0]["checksum"] == results[1]["checksum"]


def test_ddp_resumed_sidecar_advances(two_rank_resumed_run):
    results, tmpdir = two_rank_resumed_run
    assert results
    payload = torch.load(
        model_io.train_state_path(os.path.join(str(tmpdir), "cliff_ddp.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["epochs_completed"] == 3


# ---------------------------------------------------------------------------
# Contract: rank-divergent control flow
# ---------------------------------------------------------------------------


def _loop_source():
    return inspect.getsource(AM_DimerParam_Model.single_proc_train)


def _train_batches_source():
    return inspect.getsource(
        AM_DimerParam_Model._AM_DimerParam_Model__train_batches_single_proc
    )


def _evaluate_batches_source():
    return inspect.getsource(
        AM_DimerParam_Model._AM_DimerParam_Model__evaluate_batches_single_proc
    )


def test_one_loop_serves_both_launch_styles():
    """``ddp_train`` delegates; it does not carry a second epoch loop.

    A duplicated loop is how the resume sidecar, the CLIFF Eq. (23) loss and the
    induction functional version end up implemented twice and agreeing only by
    accident. It also means the golden source-introspection tests, which read
    ``single_proc_train``, would stop covering the distributed path.
    """
    source = inspect.getsource(AM_DimerParam_Model.ddp_train)
    assert "self.single_proc_train(" in source
    assert "for epoch in range(" not in source


def test_nan_early_stop_is_collective():
    """The classic DDP hang, and it looks like a job that is still training.

    One rank seeing NaN and breaking leaves the others in the next
    ``all_reduce`` until the walltime, having produced nothing.
    """
    source = _loop_source()
    assert "nan_detected" in source
    stop_index = source.index("nan_detected")
    reduce_index = source.index('op="max"', stop_index)
    break_index = source.index("break", reduce_index)
    assert reduce_index < break_index


def test_grad_norm_skip_is_collective():
    """Same shape of bug one level down: a rank that ``continue``s past a batch
    the others kept skips an optimizer step and every collective in it."""
    source = _train_batches_source()
    assert "skip_batch" in source
    skip_index = source.index("skip_batch")
    reduce_index = source.index('op="max"', skip_index)
    continue_index = source.index("continue", reduce_index)
    assert reduce_index < continue_index


def test_validation_loss_is_reduced_before_it_is_compared():
    """``lowest_test_loss`` must be compared against a *global* loss.

    Per-rank losses differ, so ranks would disagree about which epoch was best,
    save different checkpoints and resume from different sidecars.
    """
    eval_source = _evaluate_batches_source()
    assert "_ddp_reduce_epoch_metrics" in eval_source
    loop = _loop_source()
    eval_call = loop.index("__evaluate_batches_single_proc(")
    compare = loop.index("lowest_test_loss", eval_call)
    assert eval_call < compare


def test_checkpoint_and_sidecar_writes_are_rank_zero_only():
    source = _loop_source()
    assert "is_primary = rank == 0" in source
    assert "if self.model_save_path and is_primary:" in source
    assert "if train_state_file and is_primary:" in source


def test_sidecar_write_is_followed_by_a_barrier():
    """So epoch N's sidecar is on disk before any rank begins epoch N+1."""
    source = _loop_source()
    write = source.index("if train_state_file and is_primary:")
    barrier = source.index("dist.barrier()", write)
    next_epoch = source.index("track_epoch_from_locals(", write)
    assert write < barrier < next_epoch


def test_ddp_wrapper_is_rebound_into_the_dimer_models():
    """The silent-failure guard, asserted statically as well as dynamically.

    ``test_ddp_ranks_stay_bitwise_identical`` would catch a regression here, but
    only after spawning two interpreters; this says which line to look at.
    """
    source = _loop_source()
    assert "AtomTypeParam = ddp_model" in source
    assert "self.dimer_model_elst" in source[source.index("ddp_model") :]


def test_distributed_sampler_is_used_for_both_splits():
    source = _loop_source()
    assert source.count("DistributedSampler(") == 2
    assert "sampler=train_sampler" in source
    assert "sampler=test_sampler" in source
    # Validation is not shuffled, so its per-rank shards are stable across
    # epochs and the reduced loss is comparable epoch to epoch.
    assert "shuffle=False" in source


def test_batch_size_is_per_rank_and_the_global_batch_is_recorded():
    """Stated, not implied: doubling the ranks doubles the effective batch, and
    a run record that does not say so is not reproducible."""
    loop = _loop_source()
    assert "EFFECTIVE GLOBAL BATCH SIZE" in loop
    train = inspect.getsource(AM_DimerParam_Model.train)
    assert "data/effective_global_batch_size" in train
    assert "training/world_size" in train


def test_train_no_longer_refuses_multi_process():
    train = inspect.getsource(AM_DimerParam_Model.train)
    assert "NotImplementedError" not in train
    assert "mp.spawn(" in train
    assert "run_tracked_distributed(" in train
    parameters = inspect.signature(AM_DimerParam_Model.train).parameters
    assert parameters["_external_rank"].default is None
    assert parameters["_external_local_rank"].default is None


def test_single_process_path_is_untouched():
    """``world_size == 1`` must stay bitwise what it was: it is the path the
    live chunk chain is running right now."""
    loop = _loop_source()
    for guard in (
        "if world_size > 1:",
        "if world_size == 1:",
    ):
        assert guard in loop
    # Reductions only ever happen under a world_size guard, never on the
    # single-process path.
    for index in [
        i
        for i in range(len(loop))
        if loop.startswith("_ddp_reduce_epoch_metrics", i)
    ]:
        preceding = loop[:index]
        assert "if world_size > 1:" in preceding


def test_ddp_train_signature_matches_the_tracked_worker_binding():
    """``tracked_ddp_worker`` binds ``world_size``, the datasets and
    ``batch_size`` by name out of this signature, so the names are API."""
    parameters = list(
        inspect.signature(AM_DimerParam_Model.ddp_train).parameters
    )
    assert parameters[:6] == [
        "self",
        "rank",
        "world_size",
        "train_dataset",
        "test_dataset",
        "n_epochs",
    ]
    assert "batch_size" in parameters
    assert "thole_lr" in parameters
    assert "induction_diagnostics" in parameters
    assert parameters.index("thole_lr") < parameters.index("local_rank")
    assert parameters.index("induction_diagnostics") < parameters.index(
        "local_rank"
    )
    assert "local_rank" in parameters


def test_ddp_train_only_tears_down_a_group_it_created():
    """An externally launched rank's group belongs to the tracker's ``finally``.
    Destroying it twice is an error; destroying it early breaks the barrier the
    tracker still has to run."""
    source = inspect.getsource(AM_DimerParam_Model.ddp_train)
    assert "owns_process_group" in source
    assert "if owns_process_group and dist.is_initialized():" in source


def test_positive_param_routes_no_longer_force_world_size_one():
    """``train_models.py`` used to hard-code 1 here because ``train`` raised."""
    source = open(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_models.py")
    ).read()
    assert "NotImplementedError above 1" not in source
    assert "world_size = max(int(ddp_world_size or 1), 1)" in source
    # Opt-in, deliberately: a one-GPU chunk that lands on a two-GPU node must
    # not silently double its effective batch size.
    assert "torch.cuda.device_count()" in source
    device_count_index = source.index("world_size = max(int(ddp_world_size")
    assert "torch.cuda.device_count()" not in source[
        device_count_index : device_count_index + 200
    ]


def test_synthetic_split_is_picklable_into_a_spawned_rank():
    """``mp.spawn`` pickles nothing here by design, but the collate items must
    still survive a round trip or the fixture is lying about what it tests."""
    train, test = _tiny_split()
    assert len(pickle.loads(pickle.dumps(train))) == len(train)
    assert len(pickle.loads(pickle.dumps(test))) == len(test)
