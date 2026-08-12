import hashlib
from pathlib import Path

import pytest
import torch

import train_models
from apnet_pt.mace.encoder import load_verified_polar_mace
from apnet_pt.mace.model import MACEAP3D3
from apnet_pt.training.mace_ap3d3_factory import (
    _default_factory_dependencies,
    build_mace_ap3d3_harness,
    validate_mace_cli_args,
)


ARTIFACT = Path("/tmp/MACE-POLAR-1-S.model")
ARTIFACT_SHA256 = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
DATA = Path(__file__).parent / "dataset_data"
LEGACY = Path(__file__).parent / "test_models" / "ap3_ensemble_0"


def _artifact_or_skip():
    if not ARTIFACT.is_file():
        pytest.skip("local PolarMACE checkpoint is unavailable")
    actual = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    if actual != ARTIFACT_SHA256:
        pytest.skip("local PolarMACE checkpoint digest is not the pinned artifact")


def _common():
    return [
        "--mace_model_path", str(ARTIFACT),
        "--mace_model_sha256", ARTIFACT_SHA256,
        "--mace_offline",
        "--world_size_ddp", "1",
        "--omp_num_threads", "2",
        "--dataloader_num_workers", "0",
        "--skip_compile",
    ]


@pytest.mark.mace_integration
def test_real_checkpoint_atomic_heads_and_all_route_lifecycles(tmp_path):
    _artifact_or_skip()
    atomic_outputs = {}
    for mode in ("direct-completion", "learned"):
        output = tmp_path / f"atomic-{mode}.pt"
        result = train_models.main(
            [
                "--train_am", "MACE-AtomicProperties",
                "--mace_property_mode", mode,
                "--smoke_atom_data_path",
                str(DATA / "mace_atomic_properties_smoke.pkl"),
                "--n_epochs_atom", "1",
                "--am_model_path", str(output),
                *_common(),
            ]
        )
        assert output.is_file()
        assert result.lifecycle.epochs == 1
        assert result.lifecycle.reload_equal
        assert result.lifecycle.backbone_frozen
        assert result.lifecycle.gradients_finite
        atomic_outputs[mode] = output

    route_arguments = {
        "MACE-AP3D3-DirectPolar": [
            "--mace_atom_model_path", str(atomic_outputs["direct-completion"])
        ],
        "MACE-AP3D3-AtomHead": [
            "--mace_atom_model_path", str(atomic_outputs["learned"])
        ],
        "MACE-AP3D3-H1": [
            "--am_model_path", str(LEGACY / "am_3.pt"),
            "--atom_type_param_model_path", str(LEGACY / "am_h+1_3.pt"),
            "--atom_type_param_model_path2", str(LEGACY / "am_elst_h+1_3.pt"),
        ],
        "MACE-AP3D3-H2": [
            "--am_model_path", str(LEGACY / "am_3.pt"),
            "--atom_type_param_model_path", str(LEGACY / "am_h+1_3.pt"),
            "--atom_type_param_model_path2", str(LEGACY / "am_elst_h+1_3.pt"),
        ],
    }
    for option, route_args in route_arguments.items():
        output = tmp_path / f"{option}.pt"
        command_args = [
            "--train_apnet", option,
            "--smoke_data_path", str(DATA / "mace_ap3d3_smoke.pkl"),
            "--n_epochs", "1",
            "--lr", "5e-4",
            "--include_total_mse",
            "--ap_model_path", str(output),
            *route_args,
            *_common(),
        ]
        result = train_models.main(command_args)
        report = result.lifecycle
        assert output.is_file()
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        assert checkpoint["checkpoint_version"] == 3
        assert not any(
            key.startswith("featurizer.backbone.")
            for key in checkpoint["model_state_dict"]
        )
        assert report.epochs == 1
        assert report.reload_equal
        assert report.backbone_frozen
        assert report.gradients_finite
        assert report.prediction_shape[1] == 4
        assert set(report.classical_ledger) == {"elst", "indu", "disp"}
        assert set(report.residual_ledger) == {"elst", "exch", "indu", "disp"}
        assert report.long_range_classical_nonzero
        assert report.long_range_residual_zero

        if option == "MACE-AP3D3-H1":
            batch = result.dataset.test_batches[0]
            expected_prediction = result.harness.model(batch).detach()
            fresh_args = train_models.build_parser().parse_args(command_args)
            fresh_args.ap_model_path = str(tmp_path / "fresh-unused.pt")
            fresh_plan = validate_mace_cli_args(fresh_args)

            def model_factory(config, backbone):
                dependencies = _default_factory_dependencies(fresh_plan)
                fresh = build_mace_ap3d3_harness(
                    fresh_plan, dependencies=dependencies
                ).model
                fresh.featurizer.backbone = backbone
                return fresh

            def backbone_loader(path, *, map_location="cpu"):
                return load_verified_polar_mace(
                    path,
                    expected_sha256=ARTIFACT_SHA256,
                    device=map_location,
                    offline=True,
                )

            reconstructed = MACEAP3D3.load_checkpoint_v3(
                output,
                mace_artifact_path=ARTIFACT,
                model_factory=model_factory,
                backbone_loader=backbone_loader,
            )
            assert torch.equal(reconstructed(batch).detach(), expected_prediction)
