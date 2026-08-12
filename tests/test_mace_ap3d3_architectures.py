from dataclasses import FrozenInstanceError, replace

import qcelemental as qcel
import pytest
import torch

from apnet_pt.mace.schema import (
    AtomicPropertyBundle,
    MACEAtomicFeatures,
    MACEFeatureCacheKey,
    PhysicsConfig,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update,
    ap3_fused_collate_update_no_target,
    ap3_fused_collate_update_no_target_monomer_indices,
    load_hdf5_data_objects,
    qcel_dimer_to_fused_data,
    save_hdf5_data_objects,
)


def _properties(natom=2):
    return AtomicPropertyBundle(
        q=torch.zeros(natom, 1),
        mu=torch.zeros(natom, 3),
        quadrupole=torch.zeros(natom, 3, 3),
        hfvr=torch.ones(natom, 1),
        valence_width=torch.ones(natom, 1),
        alpha=torch.ones(natom, 1),
        damping=torch.ones(natom, 1),
    )


def test_physics_config_is_frozen_validated_and_hash_stable():
    first = PhysicsConfig()
    second = PhysicsConfig()

    assert first.physics_hash == second.physics_hash
    assert len(first.physics_hash) == 64
    assert first.component_order == ("elst", "exch", "indu", "disp")
    assert first.quadrupole_convention == "cartesian-symmetric-traceless-3x3"
    with pytest.raises(FrozenInstanceError):
        first.neural_cutoff = 9.0
    with pytest.raises(ValueError, match="component_order"):
        PhysicsConfig(component_order=("elst", "indu", "exch", "disp"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("electrostatics_mode", "undamped"),
        ("thole_direct", 0.41),
        ("thole_mutual", 0.42),
        ("scf_tolerance", 2.0e-8),
        ("scf_max_iterations", 201),
        ("scf_nonconvergence", "warn"),
        ("d3_parameters", (1.0, 0.9, 0.3, 3.0)),
        ("neural_cutoff", 6.0),
    ],
)
def test_each_active_physics_field_invalidates_hash(field, value):
    base = PhysicsConfig()
    changed = replace(base, **{field: value})
    assert changed.physics_hash != base.physics_hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electrostatics_parameters": [1.0]},
        {"polarizability_rule": "learned-alpha"},
        {"full_pair_edge_semantics": "cutoff-only"},
        {"thole_direct": float("nan")},
        {"thole_mutual": float("inf")},
        {"scf_tolerance": float("nan")},
        {"neural_cutoff": float("inf")},
        {"d3_parameters": [1.0, 2.0]},
        {"d3_parameters": [1.0, 2.0, 3.0, float("nan")]},
        {"component_order": ["elst", "indu", "exch", "disp"]},
        {"length_unit": "bohr"},
        {"energy_unit": "hartree"},
        {"quadrupole_convention": "spherical"},
    ],
)
def test_physics_config_rejects_inactive_or_nonfinite_semantics(kwargs):
    with pytest.raises(ValueError):
        PhysicsConfig(**kwargs)


def test_sequence_fields_are_deep_frozen_as_tuples():
    config = PhysicsConfig(
        electrostatics_parameters=[],
        d3_parameters=[1.0, 0.9, 0.3, 3.0],
        component_order=["elst", "exch", "indu", "disp"],
    )
    assert isinstance(config.electrostatics_parameters, tuple)
    assert isinstance(config.d3_parameters, tuple)
    assert isinstance(config.component_order, tuple)


def test_property_and_feature_shapes_are_validated():
    bundle = _properties()
    assert bundle.natom == 2
    with pytest.raises(ValueError, match="quadrupole"):
        AtomicPropertyBundle(
            q=torch.zeros(2, 1),
            mu=torch.zeros(2, 3),
            quadrupole=torch.zeros(2, 9),
            hfvr=torch.ones(2, 1),
            valence_width=torch.ones(2, 1),
            alpha=torch.ones(2, 1),
            damping=torch.ones(2, 1),
        )

    features = MACEAtomicFeatures(
        invariant=torch.zeros(2, 4),
        equivariant=torch.zeros(2, 3),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([1, 8]),
        total_charge=torch.tensor([0.0]),
        total_spin=torch.tensor([1.0]),
        feature_schema="polar-1-s:test",
    )
    assert features.natom == 2


def test_cache_key_includes_charge_spin_dtype_order_and_coordinates():
    base = MACEFeatureCacheKey.from_tensors(
        checkpoint_sha256="a" * 64,
        mace_version="0.3.16",
        feature_schema="polar-1-s:node_feats-512x0e",
        physics_config_hash=PhysicsConfig().physics_hash,
        atomic_numbers=torch.tensor([8, 1]),
        coordinates_angstrom=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        total_charge=0,
        total_spin=1,
        dtype=torch.float32,
    )
    moved = MACEFeatureCacheKey.from_tensors(
        checkpoint_sha256="a" * 64,
        mace_version="0.3.16",
        feature_schema="polar-1-s:node_feats-512x0e",
        physics_config_hash=PhysicsConfig().physics_hash,
        atomic_numbers=torch.tensor([8, 1]),
        coordinates_angstrom=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
        total_charge=0,
        total_spin=1,
        dtype=torch.float32,
    )
    assert base.cache_hash != moved.cache_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_sha256", "b" * 64),
        ("mace_version", "0.3.17"),
        ("feature_schema", "other-schema"),
        ("physics_config_hash", "c" * 64),
        ("atomic_numbers", (1, 8)),
        ("coordinates_angstrom", ((0.0, 0.0, 0.1), (0.0, 0.0, 1.0))),
        ("total_charge", 1.0),
        ("total_spin", 2.0),
        ("dtype", "torch.float64"),
    ],
)
def test_each_cache_key_field_invalidates_hash(field, value):
    base = MACEFeatureCacheKey.from_tensors(
        checkpoint_sha256="a" * 64,
        mace_version="0.3.16",
        feature_schema="schema",
        physics_config_hash=PhysicsConfig().physics_hash,
        atomic_numbers=torch.tensor([8, 1]),
        coordinates_angstrom=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        total_charge=0,
        total_spin=1,
        dtype=torch.float32,
    )
    assert replace(base, **{field: value}).cache_hash != base.cache_hash


def test_feature_and_property_rank_dtype_device_validation():
    with pytest.raises(ValueError, match="batch"):
        MACEAtomicFeatures(
            invariant=torch.zeros(2, 4),
            equivariant=torch.zeros(2, 3),
            batch=torch.zeros(2),
            atomic_numbers=torch.tensor([1, 8]),
            total_charge=torch.tensor([0.0]),
            total_spin=torch.tensor([1.0]),
            feature_schema="schema",
        )
    with pytest.raises(ValueError, match="floating dtype"):
        MACEAtomicFeatures(
            invariant=torch.zeros(2, 4),
            equivariant=torch.zeros(2, 3),
            batch=torch.zeros(2, dtype=torch.long),
            atomic_numbers=torch.tensor([1, 8]),
            total_charge=torch.tensor([0.0]),
            total_spin=torch.tensor([1]),
            feature_schema="schema",
        )
    values = {name: getattr(_properties(), name) for name in _properties().__dataclass_fields__}
    values["mu"] = values["mu"].double()
    with pytest.raises(ValueError, match="same dtype"):
        AtomicPropertyBundle(**values)


def test_qcel_metadata_and_full_labels_propagate_without_mutation():
    dimer = qcel.models.Molecule.from_data(
        """
        0 1
        H 0 0 0
        H 0 0 0.74
        --
        0 1
        H 0 0 4.0
        H 0 0 4.74
        units angstrom
        """
    )
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = qcel_dimer_to_fused_data(
        dimer,
        dimer_ind=0,
        r_cut=5.0,
        r_cut_im=8.0,
        y=label.clone(),
    )
    before = data.y.clone()
    batch = ap3_fused_collate_update([data])

    assert batch.total_charge_A.tolist() == [0.0]
    assert batch.total_charge_B.tolist() == [0.0]
    assert batch.total_spin_A.tolist() == [1]
    assert batch.total_spin_B.tolist() == [1]
    assert batch.batch_atomic_A.total_spin.tolist() == [1]
    assert batch.batch_atomic_B.total_spin.tolist() == [1]
    assert torch.equal(data.y, before)
    assert torch.equal(batch.y[0], label)
    assert not torch.isin(
        batch.batch_atomic_A.edge_index,
        torch.arange(batch.RA.shape[0], batch.RA.shape[0] + batch.RB.shape[0]),
    ).any()


def test_open_shell_multiplicity_propagates_through_no_target_collate():
    dimer = qcel.models.Molecule.from_data(
        """
        0 2
        H 0 0 0
        --
        0 2
        H 0 0 4.0
        units angstrom
        """
    )
    data = qcel_dimer_to_fused_data(
        dimer,
        dimer_ind=0,
        r_cut=5.0,
        r_cut_im=8.0,
    )
    batch = ap3_fused_collate_update_no_target([data])

    assert batch.total_spin_A.tolist() == [2]
    assert batch.total_spin_B.tolist() == [2]
    assert batch.batch_atomic_A.total_spin.tolist() == [2]
    assert batch.batch_atomic_B.total_spin.tolist() == [2.0]
    assert batch.batch_atomic_A.total_spin.dtype == torch.float32
    assert batch.batch_atomic_B.total_spin.dtype == torch.float32


def test_non_singlet_spin_hdf5_round_trip_and_all_collates(tmp_path):
    dimer = qcel.models.Molecule.from_data(
        """
        0 2
        H 0 0 0
        --
        0 2
        H 0 0 4.0
        units angstrom
        """
    )
    data = qcel_dimer_to_fused_data(
        dimer,
        dimer_ind=0,
        r_cut=5.0,
        r_cut_im=8.0,
        y=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    for prefix, natom in (("A", data.RA.shape[0]), ("B", data.RB.shape[0])):
        setattr(data, f"q{prefix}", torch.zeros(natom, 1))
        setattr(data, f"mu{prefix}", torch.zeros(natom, 3))
        setattr(data, f"quad{prefix}", torch.zeros(natom, 3, 3))
        setattr(data, f"hlist{prefix}", torch.zeros(natom, 2))
    path = tmp_path / "spin.h5"
    save_hdf5_data_objects([data], path)

    collates = (
        ap3_fused_collate_update,
        ap3_fused_collate_update_no_target,
        ap3_fused_collate_update_no_target_monomer_indices,
    )
    for collate in collates:
        restored = load_hdf5_data_objects(path)[0]
        batch = collate([restored])
        assert batch.total_spin_A.tolist() == [2.0]
        assert batch.total_spin_B.tolist() == [2.0]
        assert batch.batch_atomic_A.total_spin.dtype == torch.float32


def test_old_hdf5_cache_falls_back_to_singlet_float_spin(tmp_path):
    dimer = qcel.models.Molecule.from_data(
        """
        0 2
        H 0 0 0
        --
        0 2
        H 0 0 4.0
        units angstrom
        """
    )
    data = qcel_dimer_to_fused_data(dimer, dimer_ind=0)
    del data.total_spin_A
    del data.total_spin_B
    path = tmp_path / "old.h5"
    save_hdf5_data_objects([data], path)
    restored = load_hdf5_data_objects(path)[0]
    batch = ap3_fused_collate_update_no_target([restored])
    assert batch.total_spin_A.tolist() == [1.0]
    assert batch.total_spin_B.tolist() == [1.0]
    assert batch.batch_atomic_A.total_spin.dtype == torch.float32
