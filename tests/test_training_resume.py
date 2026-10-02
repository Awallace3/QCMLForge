"""An interrupted APNet2 run that resumes must reproduce the uninterrupted run.

Preemptible queues kill a job between epochs at an arbitrary point.  Resuming
from the best-model checkpoint alone restarts Adam with zero moments, restarts
the learning-rate schedule at step 0, and replays the first epoch's shuffle
order; the resume state has to carry all of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
from apnet_pt.pairwise_datasets import apnet2_module_dataset
from apnet_pt.training_resume import load_training_state

from .mols import mol_dimer

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
import train_models  # noqa: E402

N_DIMERS = 31
N_EPOCHS = 3


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    torch.manual_seed(0)
    atom_model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    # Distinct labels per dimer, so the shuffle order changes every gradient.
    labels = [
        [0.1 * i, -0.05 * i, 0.2 - 0.01 * i, 0.02 * i] for i in range(N_DIMERS)
    ]
    root = tmp_path_factory.mktemp("ap2_resume_ds")
    (root / "raw").mkdir()
    return apnet2_module_dataset(
        root=str(root),
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model=atom_model,
        atomic_batch_size=4,
        datapoint_storage_n_objects=6,
        batch_size=2,
        prebatched=False,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=0,
        qcel_molecules=[mol_dimer] * N_DIMERS,
        energy_labels=labels,
        in_memory=True,
        random_seed=None,
    )


def _train(dataset, out_dir, n_epochs, init_seed, **kwargs):
    torch.manual_seed(init_seed)
    harness = APNet2Model(use_GPU=False)
    harness.train(
        dataset,
        n_epochs=n_epochs,
        lr=5e-3,
        lr_decay=0.5,
        skip_compile=True,
        model_path=str(out_dir / "best.pt"),
        dataloader_num_workers=0,
        random_seed=7,
        checkpoint_metric="total_mae",
        resume_state_path=str(out_dir / "resume.pt"),
        wandb_config=None,
        **kwargs,
    )
    return harness


def _assert_same_tensors(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def _assert_same_optimizer(a, b):
    assert a["param_groups"] == b["param_groups"]
    assert a["state"].keys() == b["state"].keys()
    for index, slots in a["state"].items():
        for name, value in slots.items():
            assert torch.equal(value, b["state"][index][name]), (index, name)


def test_resumed_run_matches_the_uninterrupted_run(dataset, tmp_path):
    straight_dir = tmp_path / "straight"
    split_dir = tmp_path / "split"
    straight_dir.mkdir()
    split_dir.mkdir()
    straight = _train(dataset, straight_dir, N_EPOCHS, init_seed=1)

    _train(dataset, split_dir, 1, init_seed=1)
    # A different init proves the weights come from the resume state.
    resumed = _train(dataset, split_dir, N_EPOCHS, init_seed=99)

    fingerprint = None
    a = load_training_state(str(straight_dir / "resume.pt"), fingerprint)
    b = load_training_state(str(split_dir / "resume.pt"), fingerprint)
    assert a["epochs_completed"] == b["epochs_completed"] == N_EPOCHS
    assert a["n_epochs"] == b["n_epochs"] == N_EPOCHS
    assert a["best_epoch"] == b["best_epoch"]
    assert a["best_score"] == b["best_score"]
    _assert_same_tensors(a["model_state_dict"], b["model_state_dict"])
    _assert_same_tensors(a["best_model_state_dict"], b["best_model_state_dict"])
    _assert_same_optimizer(a["optimizer_state_dict"], b["optimizer_state_dict"])
    assert a["scheduler_state_dict"] == b["scheduler_state_dict"]
    # Both runs finish holding the same best weights.
    _assert_same_tensors(straight.model.state_dict(), resumed.model.state_dict())


def test_a_finished_state_trains_no_further_epochs(dataset, tmp_path):
    _train(dataset, tmp_path, 1, init_seed=1)
    before = load_training_state(str(tmp_path / "resume.pt"), None)
    again = _train(dataset, tmp_path, 1, init_seed=99)
    after = load_training_state(str(tmp_path / "resume.pt"), None)
    assert after["epochs_completed"] == 1
    _assert_same_tensors(before["model_state_dict"], after["model_state_dict"])
    _assert_same_tensors(
        before["best_model_state_dict"], again.model.state_dict()
    )


def test_resume_rejects_a_changed_training_setup(dataset, tmp_path):
    """Continuing under another lr or selection metric is a different run."""
    _train(dataset, tmp_path, 1, init_seed=1)
    with pytest.raises(ValueError, match="lr"):
        torch.manual_seed(1)
        APNet2Model(use_GPU=False).train(
            dataset,
            n_epochs=2,
            lr=1e-3,
            lr_decay=0.5,
            skip_compile=True,
            dataloader_num_workers=0,
            random_seed=7,
            checkpoint_metric="total_mae",
            resume_state_path=str(tmp_path / "resume.pt"),
        )


def test_resume_is_refused_for_multi_process_training(dataset, tmp_path):
    with pytest.raises(ValueError, match="single-process"):
        APNet2Model(use_GPU=False).train(
            dataset,
            n_epochs=1,
            world_size=2,
            resume_state_path=str(tmp_path / "resume.pt"),
        )


def test_cli_accepts_a_resume_state_path():
    args = train_models.build_arg_parser().parse_args(
        ["--resume-state", "/scratch/run/resume.pt"]
    )
    assert args.resume_state == "/scratch/run/resume.pt"
    assert train_models.build_arg_parser().parse_args([]).resume_state is None


def test_cli_refuses_resume_on_a_route_that_cannot_resume():
    """Dropping the flag silently would restart a preempted run from scratch."""

    def train(model_path=None, n_epochs=1):
        pass

    with pytest.raises(ValueError, match="cannot resume"):
        train_models.resume_state_train_kwargs("APNet3", train, "/tmp/r.pt")
    assert train_models.resume_state_train_kwargs("APNet3", train, None) == {}


def test_resume_state_loads_without_unpickling_arbitrary_objects(dataset, tmp_path):
    _train(dataset, tmp_path, 1, init_seed=3)
    state = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=True)
    assert state["epochs_completed"] == 1


def test_default_criterion_is_the_mse_loss_it_always_was(dataset, tmp_path):
    """No ``loss_fn`` must train exactly as an explicit ``torch.nn.MSELoss()``."""
    weights = []
    for name, loss_fn in (("default", None), ("explicit", torch.nn.MSELoss())):
        out_dir = tmp_path / name
        out_dir.mkdir()
        harness = _train(dataset, out_dir, 2, init_seed=5, loss_fn=loss_fn)
        weights.append(harness.model.state_dict())
    _assert_same_tensors(*weights)


def test_transfer_learning_refuses_a_selected_loss(dataset, tmp_path):
    with pytest.raises(ValueError, match="transfer_learning"):
        APNet2Model(use_GPU=False).train(
            dataset,
            n_epochs=1,
            transfer_learning=True,
            loss_fn=torch.nn.MSELoss(),
        )


def test_a_v1_pickled_state_is_refused_with_an_explanation(tmp_path):
    """v1 pickled NumPy RNG state, which ``weights_only=True`` cannot read."""
    path = tmp_path / "resume.pt"
    torch.save({"format": "qcmlforge-training-resume-v1", "rng": object()}, path)
    with pytest.raises(ValueError, match="source commit that wrote it"):
        load_training_state(str(path), None)
