import builtins
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import warnings

import pytest
import torch

import train_models

from apnet_pt.training.mace_ap3d3_factory import (
    MACE_AP3D3_OPTIONS,
    MACEFactoryDependencies,
    build_mace_ap3d3_harness,
    build_mace_atomic_harness,
    dispatch_mace_cli,
    expected_resume_semantics,
    resolve_mace_option,
    validate_mace_cli_args,
)
from tests.test_mace_model_harness import (
    StubFeaturizer,
    StubLongRangeProvider,
    StubPropertyProvider,
    _make_model,
)


CANONICAL = {
    "MACE-AP3D3-DirectPolar": {
        "properties": "direct",
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
    },
    "MACE-AP3D3-H1": {
        "properties": "legacy",
        "pair_mode": "h1",
        "feature_mode": "final-layer-scalars",
    },
    "MACE-AP3D3-H2": {
        "properties": "legacy",
        "pair_mode": "h2",
        "feature_mode": "all-scalars+norms",
    },
    "MACE-AP3D3-AtomHead": {
        "properties": "atomhead",
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
    },
}


def _write(path, content=b"checkpoint"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _base_cli(tmp_path, option="MACE-AP3D3-H1"):
    mace_path = tmp_path / "mace.model"
    _write(mace_path, b"mace-foundation")
    mace_sha = hashlib.sha256(mace_path.read_bytes()).hexdigest()
    paths = {
        "am": _write(tmp_path / "am.pt"),
        "p1": _write(tmp_path / "param1.pt"),
        "p2": _write(tmp_path / "param2.pt"),
        "atomhead": _write(tmp_path / "atomhead.pt"),
        "data": str(
            Path(__file__).parent / "dataset_data" / "mace_ap3d3_smoke.pkl"
        ),
    }
    argv = [
        "--train_apnet",
        option,
        "--ap_model_path",
        str(tmp_path / "output.pt"),
        "--mace_model_path",
        str(mace_path),
        "--mace_model_sha256",
        mace_sha,
        "--smoke_data_path",
        paths["data"],
        "--world_size_ddp",
        "3",
        "--n_epochs",
        "1",
        "--skip_compile",
    ]
    if option in {"MACE-AP3D3-H1", "MACE-AP3D3-H2"}:
        argv += [
            "--am_model_path",
            paths["am"],
            "--atom_type_param_model_path",
            paths["p1"],
            "--atom_type_param_model_path2",
            paths["p2"],
        ]
    else:
        argv += ["--mace_atom_model_path", paths["atomhead"]]
    return train_models.build_parser().parse_args(argv), paths


def test_registry_has_exact_case_sensitive_canonical_options():
    assert MACE_AP3D3_OPTIONS == CANONICAL
    for name, expected in CANONICAL.items():
        resolved = resolve_mace_option(name)
        assert resolved.public_name == name
        assert resolved.properties == expected["properties"]
        assert resolved.pair_mode == expected["pair_mode"]
        assert resolved.feature_mode == expected["feature_mode"]
        assert resolved.run_name == name
        assert not resolved.is_ablation
    for wrong in ("mace-ap3d3-h1", "MACE-AP3D3-h1", "MACE-AP3D3-Unknown"):
        with pytest.raises(ValueError, match="exact case-sensitive"):
            resolve_mace_option(wrong)


def test_noncanonical_feature_override_is_named_ablation():
    resolved = resolve_mace_option(
        "MACE-AP3D3-H1", feature_mode="all-scalars+norms"
    )
    assert resolved.is_ablation
    assert resolved.run_name == (
        "MACE-AP3D3-H1__ablation-feature-all-scalars+norms"
    )
    assert resolved.public_name == "MACE-AP3D3-H1"
    with pytest.raises(ValueError, match="feature mode"):
        resolve_mace_option("MACE-AP3D3-H1", feature_mode="vectors")


def test_parser_exposes_complete_mace_and_retained_legacy_contract():
    parser = train_models.build_parser()
    args = parser.parse_args(
        [
            "--train_am",
            "MACE-AtomicProperties",
            "--mace_model",
            "polar-1-s",
            "--mace_model_path",
            "/tmp/mace",
            "--mace_model_sha256",
            "a" * 64,
            "--mace_feature_mode",
            "all-scalars+norms",
            "--mace_default_dtype",
            "float64",
            "--mace_cache_dir",
            "/tmp/cache",
            "--mace_offline",
            "--mace_atom_model_path",
            "/tmp/atom",
            "--mace_property_mode",
            "direct-completion",
            "--train_atomic_heads",
            "--long_range_elst",
            "undamped",
            "--d3_params",
            "1,2,3,4",
            "--smoke_data_path",
            "/tmp/dimers",
            "--smoke_atom_data_path",
            "/tmp/atoms",
            "--skip_compile",
            "--dataloader_num_workers",
            "0",
            "--overwrite",
            "--no_disp_nn",
            "--include_total_mse",
            "--no-use_precomputed_classical",
            "--unfreeze_dimer_prop_model",
            "--unfreeze_atom_model",
            "--build_dataset_only",
            "--world_size_ddp",
            "4",
            "--omp_num_threads",
            "2",
        ]
    )
    for name in (
        "mace_model",
        "mace_model_path",
        "mace_model_sha256",
        "mace_feature_mode",
        "mace_default_dtype",
        "mace_cache_dir",
        "mace_offline",
        "mace_atom_model_path",
        "mace_property_mode",
        "train_atomic_heads",
        "long_range_elst",
        "d3_params",
        "smoke_data_path",
        "smoke_atom_data_path",
        "skip_compile",
        "dataloader_num_workers",
        "overwrite",
        "resume",
    ):
        assert hasattr(args, name)
    assert args.world_size_ddp == 4
    assert args.use_precomputed_classical is False
    assert args.no_disp_nn and args.include_total_mse
    assert args.unfreeze_dimer_prop_model and args.unfreeze_atom_model


@pytest.mark.parametrize("option", CANONICAL)
def test_all_routes_validate_before_dataset_and_preserve_runtime_flags(
    tmp_path, option
):
    args, _ = _base_cli(tmp_path, option)
    args.no_disp_nn = True
    args.include_total_mse = True
    args.use_precomputed_classical = False
    args.unfreeze_dimer_prop_model = True
    args.unfreeze_atom_model = True
    args.build_dataset_only = True
    args.skip_compile = True
    args.dataloader_num_workers = 0
    plan = validate_mace_cli_args(args)
    expected = CANONICAL[option]
    assert plan.properties == expected["properties"]
    assert plan.pair_mode == expected["pair_mode"]
    assert plan.feature_mode == expected["feature_mode"]
    assert plan.world_size_ddp == 3
    assert plan.no_disp_nn and plan.include_total_mse
    assert plan.use_precomputed_classical is False
    assert plan.unfreeze_dimer_prop_model and plan.unfreeze_atom_model
    assert plan.build_dataset_only and plan.skip_compile
    assert plan.dataloader_num_workers == 0
    assert plan.resume is False


@pytest.mark.parametrize(
    ("option", "mutation", "message"),
    [
        ("MACE-AP3D3-H1", {"atom_type_param_model_path": None}, "all three"),
        ("MACE-AP3D3-H2", {"mace_atom_model_path": "bad.pt"}, "rejects"),
        ("MACE-AP3D3-DirectPolar", {"mace_atom_model_path": None}, "requires"),
        ("MACE-AP3D3-AtomHead", {"mace_atom_model_path": None}, "requires"),
    ],
)
def test_route_validation_fails_before_dataset_builder(
    tmp_path, option, mutation, message
):
    args, _ = _base_cli(tmp_path, option)
    for key, value in mutation.items():
        setattr(args, key, value)
    calls = []
    dependencies = MACEFactoryDependencies(dataset_builder=lambda plan: calls.append(plan))
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        dispatch_mace_cli(args, dependencies=dependencies)
    assert calls == []


def test_wrong_case_fails_before_any_factory_or_dataset_builder(tmp_path):
    args, _ = _base_cli(tmp_path)
    args.train_apnet = "mace-ap3d3-h1"
    calls = []
    with pytest.raises(ValueError, match="exact case-sensitive"):
        dispatch_mace_cli(
            args,
            dependencies=MACEFactoryDependencies(
                featurizer_builder=lambda plan: calls.append("model"),
                dataset_builder=lambda plan: calls.append("dataset"),
            ),
        )
    assert calls == []


def test_train_atomic_heads_without_labels_warns_and_mace_stays_frozen(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-AtomHead")
    args.mace_atom_model_path = None
    args.train_atomic_heads = True
    args.smoke_atom_data_path = None
    args.unfreeze_atom_model = True
    with pytest.warns(RuntimeWarning, match="code-validation only"):
        plan = validate_mace_cli_args(args)
    assert plan.train_atomic_heads
    assert plan.freeze_mace_backbone


def test_checked_smoke_fixture_hashes_replace_database_split_hashes(tmp_path):
    args, paths = _base_cli(tmp_path)
    with Path(paths["data"]).open("rb") as handle:
        fixture = pickle.load(handle)
    plan = validate_mace_cli_args(args)
    assert plan.data_hash == fixture["content_hash"]
    assert plan.preprocessing_hash == fixture["preprocessing_hash"]
    assert plan.split_hash == fixture["split_hash"]


def test_visible_gpus_do_not_override_user_world_size(tmp_path, monkeypatch):
    args, _ = _base_cli(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    plan = validate_mace_cli_args(args)
    assert plan.world_size_ddp == 3


def test_existing_output_requires_overwrite_or_explicit_resume(tmp_path):
    args, _ = _base_cli(tmp_path)
    Path(args.ap_model_path).write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite.*resume"):
        validate_mace_cli_args(args)
    args.overwrite = True
    plan = validate_mace_cli_args(args)
    assert plan.overwrite and not plan.resume
    args.resume = True
    with pytest.raises(ValueError, match="resume.*not implemented"):
        validate_mace_cli_args(args)


def test_resume_is_rejected_before_dataset_or_model_build(tmp_path):
    args, _ = _base_cli(tmp_path)
    args.resume = True
    calls = []
    with pytest.raises(ValueError, match="resume.*not implemented"):
        dispatch_mace_cli(
            args,
            dependencies=MACEFactoryDependencies(
                featurizer_builder=lambda plan: calls.append(plan),
                dataset_builder=lambda plan: calls.append(plan),
            ),
        )
    assert calls == []


def test_factory_builds_all_four_injected_harnesses(tmp_path):
    for option in CANONICAL:
        args, _ = _base_cli(tmp_path / option.replace("-", "_"), option)
        args.include_total_mse = option == "MACE-AP3D3-H1"
        Path(args.mace_model_path).parent.mkdir(parents=True, exist_ok=True)
        # _base_cli wrote before nested parent existed in old pathlib versions.
        plan = validate_mace_cli_args(args)
        reference, _, _ = _make_model(plan.internal_architecture)
        dependencies = MACEFactoryDependencies(
            featurizer_builder=lambda plan, value=reference.featurizer: value,
            property_provider_builder=(
                lambda plan, featurizer, value=reference.property_provider: value
            ),
            pair_core_builder=lambda plan, value=reference.pair_core: value,
            long_range_builder=(
                lambda plan, value=reference.long_range_provider: value
            ),
        )
        harness = build_mace_ap3d3_harness(plan, dependencies=dependencies)
        assert harness.model.architecture == plan.internal_architecture
        assert harness.model.featurizer.backbone.training is False
        assert harness.include_total_mse is plan.include_total_mse


def test_atomic_factory_builds_injected_harness(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-AtomHead")
    args.train_apnet = ""
    args.train_am = "MACE-AtomicProperties"
    args.am_model_path = str(tmp_path / "atomic-output.pt")
    args.train_atomic_heads = True
    args.smoke_atom_data_path = None
    plan = validate_mace_cli_args(args)
    featurizer = StubFeaturizer("all-scalars+norms")
    provider = StubPropertyProvider("atomhead")
    dependencies = MACEFactoryDependencies(
        featurizer_builder=lambda plan: featurizer,
        property_provider_builder=lambda plan, features: provider,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harness = build_mace_atomic_harness(plan, dependencies=dependencies)
    assert harness.property_mode == "learned"
    assert all(not p.requires_grad for p in harness.featurizer.backbone.parameters())


def test_mace_training_modules_do_not_eagerly_import_optional_mace(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mace" or name.startswith("mace."):
            raise AssertionError("optional MACE import crossed lazy boundary")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    import apnet_pt.training.mace_ap3d3_factory as factory

    assert set(factory.MACE_AP3D3_OPTIONS) == set(CANONICAL)


def test_train_models_legacy_dispatch_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(train_models, "train_pairwise_model", lambda **kw: calls.append(kw))
    train_models.main(["--train_apnet", "APNet2", "--world_size_ddp", "7"])
    assert calls[0]["apnet_model_type"] == "APNet2"
    assert calls[0]["model_out"] == "./models/ap_default.pt"
