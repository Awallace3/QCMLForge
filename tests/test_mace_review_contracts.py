"""Correctness and scientific-contract regressions from final review pass A."""

from dataclasses import replace
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from apnet_pt import constants
from apnet_pt.mace.long_range import LongRangeSAPTProvider
from apnet_pt.mace.model import MACEAP3D3Model
from apnet_pt.mace.properties import (
    LegacyAtomMPNNPropertyProvider,
    PolarDirectPropertyProvider,
)
from apnet_pt.mace.schema import (
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    InductionDiagnostics,
    MACEAtomicFeatures,
    PhysicsConfig,
    PolarMACEDirectOutputs,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update_no_target,
    qcel_dimer_to_fused_data,
)
from apnet_pt.training.mace_ap3d3_factory import (
    MACEFactoryDependencies,
    build_mace_ap3d3_harness,
    validate_mace_cli_args,
)
from apnet_pt.training.smoke import (
    load_pair_smoke_fixture,
    load_prepared_feature_cache,
)
from tests.test_mace_ap3d3_cli import _base_cli
from tests.test_mace_atomic_properties import _direct_outputs, _features, _heads
from tests.test_mace_model_harness import _make_model


DATA = Path(__file__).parent / "dataset_data"
PAIR_FIXTURE = DATA / "mace_ap3d3_smoke.pkl"
ATOM_FIXTURE = DATA / "mace_atomic_properties_smoke.pkl"


def _assert_primitive(value):
    assert not type(value).__module__.startswith("qcelemental")
    if isinstance(value, dict):
        for item in value.values():
            _assert_primitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_primitive(item)


@pytest.mark.parametrize("path", [PAIR_FIXTURE, ATOM_FIXTURE])
def test_smoke_pickles_contain_only_version_neutral_molecule_records(path):
    raw = path.read_bytes()
    assert b"qcelemental.models.v1" not in raw
    assert b"qcelemental.models.molecule" not in raw
    fixture = pickle.loads(raw)
    _assert_primitive(fixture)
    molecule_key = "dimer" if path == PAIR_FIXTURE else "monomer"
    for record in fixture["records"]:
        molecule = record[molecule_key]
        assert molecule["format"] == "qcel-psi4-text-v1"
        assert isinstance(molecule["data"], str)


def test_directpolar_keeps_public_charges_and_rejects_centered_total_mismatch():
    positions = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    charges = torch.tensor([0.2, 0.2])
    intrinsic = torch.zeros(2, 3)
    direct = _direct_outputs(positions, charges, intrinsic, total_charge=0.4)
    features = MACEAtomicFeatures(
        invariant=torch.randn(2, 5),
        equivariant=torch.randn(2, 18),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([1, 8]),
        total_charge=torch.tensor([0.4]),
        total_spin=torch.tensor([1.0]),
        feature_schema="stub:all-scalars+norms:equiv=2x0e+2x1o+2x2e",
    )
    bundle = PolarDirectPropertyProvider(_heads()).forward_monomer(features, direct)
    assert torch.equal(bundle.q[:, 0], direct.charges)

    inconsistent_features = replace(features, total_charge=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="direct.*total_charge|total charges"):
        PolarDirectPropertyProvider(_heads()).forward_monomer(
            inconsistent_features, direct
        )

    inconsistent_direct = PolarMACEDirectOutputs(
        density_coefficients=direct.density_coefficients,
        charges=direct.charges,
        molecular_dipole_eangstrom=direct.molecular_dipole_eangstrom,
        positions_angstrom=direct.positions_angstrom,
        batch=direct.batch,
        total_charge=torch.tensor([0.0]),
    )
    with pytest.raises(ValueError, match="charges.*total_charge"):
        PolarDirectPropertyProvider(_heads()).forward_monomer(
            features, inconsistent_direct
        )


def test_plan_hashes_real_residual_cutoff_and_rejects_conflict_before_build(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    plan = validate_mace_cli_args(args)
    assert plan.neural_cutoff == plan.r_cut_im == 8.0
    assert PhysicsConfig(neural_cutoff=plan.r_cut_im).physics_hash == plan.physics_hash

    args, _ = _base_cli(tmp_path / "conflict", "MACE-AP3D3-H1")
    args.r_cut_im = 7.5
    with pytest.raises(ValueError, match="r_cut_im|neural cutoff"):
        validate_mace_cli_args(args)


def test_prepared_cache_rejects_neural_cutoff_physics_hash_mismatch(tmp_path):
    old = PhysicsConfig(neural_cutoff=7.0)
    current = PhysicsConfig(neural_cutoff=8.0)
    (tmp_path / "COMPLETE.json").write_text(
        __import__("json").dumps({
            "status": "complete",
            "cache_format": "qcmlforge-mace-monomer-cache-v1",
            "mace_sha256": "a" * 64,
            "mace_model_id": "polar-1-s",
            "physics_hash": old.physics_hash,
            "dtype": "float32",
            "dataset_hash": "b" * 64,
            "preprocessing_hash": "c" * 64,
            "split_hash": "d" * 64,
            "feature_schemas": {},
            "entry_count": 0,
            "entries": [],
        })
    )
    with pytest.raises(RuntimeError, match="cache mismatch for physics_hash"):
        load_prepared_feature_cache(
            tmp_path,
            feature_mode="final-layer-scalars",
            mace_sha256="a" * 64,
            mace_model_id="polar-1-s",
            physics_hash=current.physics_hash,
            dataset_kind="pair",
            dataset_hash="b" * 64,
            preprocessing_hash="c" * 64,
            split_hash="d" * 64,
            dtype=torch.float32,
        )


def _canonical_props(numbers):
    hfvr = torch.ones(len(numbers), 1)
    alpha = constants.polarizability_table[numbers].reshape(-1, 1) * hfvr.pow(4 / 3)
    return AtomicPropertyBundle(
        q=torch.tensor([[-0.4], [0.2], [0.2]]),
        mu=torch.zeros(3, 3),
        quadrupole=torch.zeros(3, 3, 3),
        hfvr=hfvr,
        valence_width=torch.ones(3, 1),
        alpha=alpha,
        damping=torch.ones(3, 1),
    )


def test_long_range_rejects_alpha_inconsistent_with_canonical_hfvr_rule():
    from tests.test_long_range_sapt import _batch, _props

    batch = _batch()
    props = replace(_props(), alpha=torch.ones(2, 1))
    with pytest.raises(ValueError, match="alpha.*HFVR|canonical alpha"):
        LongRangeSAPTProvider(
            PhysicsConfig(electrostatics_mode="undamped"),
            electrostatics_kernels={"undamped": lambda **kwargs: torch.zeros(2)},
            induction_kernel=lambda **kwargs: (
                torch.zeros(2),
                {"converged": True, "iterations": 1, "residual": 0.0},
            ),
            dispersion_kernel=lambda batch, params: torch.zeros(2),
        )(batch, props, props)


def test_amoeba_hippo_reference_uses_explicit_k_and_production_provider():
    reference = pd.read_pickle(DATA / "amoeba_water_dimer_ref.pkl")
    batch = ap3_fused_collate_update_no_target(
        [qcel_dimer_to_fused_data(
            reference["qcel_molecule"], r_cut_im=99.0, dimer_ind=0
        )]
    )
    batch.amoeba_K_A = torch.tensor(reference["alpha_A"], dtype=torch.float32)
    batch.amoeba_K_B = torch.tensor(reference["alpha_B"], dtype=torch.float32)
    props_a = _canonical_props(batch.ZA)
    props_b = _canonical_props(batch.ZB)
    props_a = replace(
        props_a,
        q=torch.tensor(reference["q_A pbe0/atz"], dtype=torch.float32).reshape(-1, 1),
        mu=torch.tensor(reference["mu_A pbe0/atz"], dtype=torch.float32),
        quadrupole=torch.tensor(reference["theta_A pbe0/atz"], dtype=torch.float32),
    )
    props_b = replace(
        props_b,
        q=torch.tensor(reference["q_B pbe0/atz"], dtype=torch.float32).reshape(-1, 1),
        mu=torch.tensor(reference["mu_B pbe0/atz"], dtype=torch.float32),
        quadrupole=torch.tensor(reference["theta_B pbe0/atz"], dtype=torch.float32),
    )
    result = LongRangeSAPTProvider(PhysicsConfig(electrostatics_mode="damped-amoeba"))(
        batch, props_a, props_b
    )
    # q is the total atomic monopole; the production kernel subtracts Z. This
    # frozen production-path value uses HIPPO K but is intentionally not called
    # the independent HIPPO energy: the two values do not agree scientifically.
    assert torch.allclose(
        result.dimer_elst,
        torch.tensor([-5.707708358764648]),
        atol=2.0e-5,
        rtol=2.0e-6,
    )
    assert abs(
        float(result.dimer_elst) - float(reference["amoeba_elst_hippo"])
    ) > 1.0

    del batch.amoeba_K_A
    with pytest.raises(ValueError, match="AMOEBA.*damping"):
        LongRangeSAPTProvider(PhysicsConfig(electrostatics_mode="damped-amoeba"))(
            batch, props_a, props_b
        )


def test_precomputed_classical_exact_prediction_and_loss_parity():
    model, _, _ = _make_model("hybrid-h1")
    dataset = load_pair_smoke_fixture(PAIR_FIXTURE)
    batch = dataset.train_batches[0]
    live_details = model(batch, return_details=True)
    batch.precomputed_classical = live_details.classical
    model.use_precomputed_classical = True
    precomputed_details = model(batch, return_details=True)
    assert torch.equal(live_details.components, precomputed_details.components)
    assert torch.equal(live_details.residual, precomputed_details.residual)
    assert precomputed_details.classical.physics_config_hash == dataset.physics_hash

    harness = MACEAP3D3Model(model, include_total_mse=True)
    precomputed_loss, _ = harness.compute_loss(batch)
    model.use_precomputed_classical = False
    live_loss, _ = harness.compute_loss(batch)
    assert torch.equal(live_loss, precomputed_loss)

    batch.precomputed_classical = replace(
        batch.precomputed_classical, physics_config_hash="0" * 64
    )
    model.use_precomputed_classical = True
    with pytest.raises(ValueError, match="precomputed classical.*physics"):
        model(batch)


def test_requested_precomputed_smoke_without_ledgers_fails_before_build(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    args.use_precomputed_classical = True
    called = False

    def builder(plan):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="precomputed classical.*ledgers"):
        validate_mace_cli_args(args)
    assert not called


class _SelectiveLegacy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_model = torch.nn.Linear(2, 2)
        self.dimer_head = torch.nn.Linear(2, 2)


@pytest.mark.parametrize("unfreeze_atom", [False, True])
@pytest.mark.parametrize("unfreeze_dimer", [False, True])
def test_legacy_selective_unfreeze_combinations(unfreeze_atom, unfreeze_dimer):
    legacy = _SelectiveLegacy()
    provider = LegacyAtomMPNNPropertyProvider(
        legacy,
        freeze_atom_model=not unfreeze_atom,
        freeze_dimer_parameters=not unfreeze_dimer,
    )
    assert all(
        parameter.requires_grad is unfreeze_atom
        for parameter in legacy.atom_model.parameters()
    )
    assert all(
        parameter.requires_grad is unfreeze_dimer
        for parameter in legacy.dimer_head.parameters()
    )

    model, _, _ = _make_model("hybrid-h1")
    model.property_provider = provider
    assert all(not p.requires_grad for p in model.featurizer.backbone.parameters())
