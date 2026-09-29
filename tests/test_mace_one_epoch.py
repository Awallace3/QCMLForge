from pathlib import Path

import pytest
import torch

import train_models
from apnet_pt.mace.model import MACEAtomicPropertiesModel
from apnet_pt.training.mace_ap3d3_factory import (
    MACEFactoryDependencies,
    dispatch_mace_cli,
)
from apnet_pt.training.smoke import (
    load_atomic_smoke_fixture,
    load_pair_smoke_fixture,
    run_atomic_smoke_lifecycle,
    run_pair_smoke_lifecycle,
)
from tests.test_mace_ap3d3_cli import _base_cli
from tests.test_mace_model_harness import (
    StubFeaturizer,
    StubPropertyProvider,
    _make_model,
)


DATA = Path(__file__).parent / "dataset_data"
PAIR_FIXTURE = DATA / "mace_ap3d3_smoke.pkl"
ATOM_FIXTURE = DATA / "mace_atomic_properties_smoke.pkl"
PUBLIC_ROUTES = {
    "MACE-AP3D3-DirectPolar": "direct-polar",
    "MACE-AP3D3-H1": "hybrid-h1",
    "MACE-AP3D3-H2": "hybrid-h2",
    "MACE-AP3D3-AtomHead": "atomhead",
}


@pytest.mark.parametrize(
    ("property_mode", "provider_kind"),
    [("direct-completion", "direct"), ("learned", "atomhead")],
)
def test_atomic_modes_one_epoch_checkpoint_and_reload(
    tmp_path, property_mode, provider_kind
):
    torch.manual_seed(101)
    dataset = load_atomic_smoke_fixture(ATOM_FIXTURE)
    model = MACEAtomicPropertiesModel(
        property_mode=property_mode,
        featurizer=StubFeaturizer("all-scalars+norms"),
        property_provider=StubPropertyProvider(provider_kind),
    )
    output = tmp_path / f"atomic-{property_mode}.pt"
    report = run_atomic_smoke_lifecycle(
        model,
        dataset,
        output_path=output,
        learning_rate=1.0e-3,
    )

    assert report.epochs == 1
    assert output.is_file()
    assert report.reload_equal
    assert report.backbone_frozen
    assert report.gradients_finite
    assert torch.isfinite(torch.tensor(report.loss))
    assert set(report.losses) == {
        "q", "mu", "quadrupole", "hfvr", "valence_width", "alpha", "damping"
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in report.losses.values())


@pytest.mark.parametrize(("public_name", "route"), PUBLIC_ROUTES.items())
def test_all_pair_routes_one_epoch_ledgers_cutoff_checkpoint_reload(
    tmp_path, public_name, route
):
    torch.manual_seed(103)
    dataset = load_pair_smoke_fixture(PAIR_FIXTURE, batch_size=16)
    model, _, _ = _make_model(route)
    output = tmp_path / f"{public_name}.pt"
    report = run_pair_smoke_lifecycle(
        model,
        dataset,
        output_path=output,
        learning_rate=1.0e-3,
        include_total_mse=True,
    )

    assert report.epochs == 1
    assert output.is_file()
    assert report.prediction_shape[1] == 4
    assert report.reload_equal
    assert report.backbone_frozen
    assert report.gradients_finite
    assert torch.isfinite(torch.tensor(report.loss))
    assert set(report.component_losses) == {"elst", "exch", "indu", "disp"}
    assert set(report.classical_ledger) == {"elst", "indu", "disp"}
    assert set(report.residual_ledger) == {"elst", "exch", "indu", "disp"}
    assert report.long_range_classical_nonzero
    assert report.long_range_residual_zero
    assert report.induction_converged
    assert report.induction_iterations >= 0
    assert report.induction_residual >= 0.0
    assert report.induction_policy in {"raise", "warn"}
    diagnostics = Path(f"{output}.diagnostics.json")
    assert diagnostics.is_file()
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert "induction_diagnostics" in checkpoint["config"]


def test_documented_train_models_entry_runs_smoke_lifecycle(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    args.n_epochs = 1
    args.skip_compile = True
    args.dataloader_num_workers = 0
    model, _, _ = _make_model("hybrid-h1")
    dependencies = MACEFactoryDependencies(
        featurizer_builder=lambda plan: model.featurizer,
        property_provider_builder=lambda plan, featurizer: model.property_provider,
        pair_core_builder=lambda plan: model.pair_core,
        long_range_builder=lambda plan: model.long_range_provider,
        model_builder=lambda plan, featurizer, provider, pair, long_range: model,
        dataset_builder=lambda plan: load_pair_smoke_fixture(plan.smoke_data_path),
        lifecycle_runner=lambda plan, harness, dataset: run_pair_smoke_lifecycle(
            harness.model,
            dataset,
            output_path=plan.output_path,
            learning_rate=plan.learning_rate,
            include_total_mse=plan.include_total_mse,
        ),
    )
    result = train_models.dispatch_args(
        args,
        mace_dispatch=lambda parsed: dispatch_mace_cli(
            parsed, dependencies=dependencies
        ),
    )
    assert result.lifecycle.epochs == 1
    assert result.lifecycle.prediction_shape[1] == 4
    assert Path(args.ap_model_path).is_file()


def test_matched_baseline_uses_identical_fixture_split_and_physics(tmp_path):
    dataset = load_pair_smoke_fixture(PAIR_FIXTURE, batch_size=16)
    legacy = Path(__file__).parent / "test_models" / "ap3_ensemble_0"
    output = tmp_path / "apnet3-fused-d3-baseline.pt"
    report = train_models.main(
        [
            "--train_apnet", "APNet3-fused-d3",
            "--am_model_path", str(legacy / "am_3.pt"),
            "--atom_type_param_model_path", str(legacy / "am_h+1_3.pt"),
            "--atom_type_param_model_path2", str(legacy / "am_elst_h+1_3.pt"),
            "--smoke_data_path", str(PAIR_FIXTURE),
            "--n_epochs", "1",
            "--lr", "1e-3",
            "--world_size_ddp", "1",
            "--dataloader_num_workers", "0",
            "--skip_compile",
            "--include_total_mse",
            "--ap_model_path", str(output),
        ]
    )
    assert output.is_file()
    assert report.baseline_name == "APNet3-fused-d3"
    assert report.split_hash == dataset.split_hash
    assert report.physics_hash == dataset.physics_hash
    assert report.prediction_shape[1] == 4
    assert report.reload_equal
    assert report.gradients_finite
    assert set(report.classical_ledger) == {"elst", "indu", "disp"}
    assert set(report.residual_ledger) == {"elst", "exch", "indu", "disp"}
    assert report.long_range_classical_nonzero
    assert report.long_range_residual_zero
