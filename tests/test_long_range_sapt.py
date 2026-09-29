import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    induced_dipole_induction_optimized_no_correction,
)
from apnet_pt.mace.long_range import LongRangeSAPTProvider, assemble_sapt_components
from apnet_pt.mace.schema import AtomicPropertyBundle, PhysicsConfig
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update_no_target,
    dimer_fused_data,
    qcel_dimer_to_fused_data,
)


def _batch():
    return SimpleNamespace(
        ZA=torch.tensor([1, 8]),
        RA=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ZB=torch.tensor([1, 8]),
        RB=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
        e_AA_source=torch.tensor([0, 1]),
        e_AA_target=torch.tensor([1, 0]),
        e_BB_source=torch.tensor([0, 1]),
        e_BB_target=torch.tensor([1, 0]),
        e_ABfull_source=torch.tensor([0, 1]),
        e_ABfull_target=torch.tensor([0, 1]),
        dimer_ind_full=torch.tensor([0, 0]),
        total_charge_A=torch.tensor([0.0]),
        total_charge_B=torch.tensor([0.0]),
    )


def _alpha(numbers, hfvr):
    return (
        constants.polarizability_table[torch.as_tensor(numbers, dtype=torch.long)]
        .reshape(-1, 1)
        .to(hfvr)
        * hfvr.abs().pow(4.0 / 3.0)
    )


def _props():
    hfvr = torch.ones(2, 1)
    return AtomicPropertyBundle(
        q=torch.tensor([[0.4], [-0.4]]),
        mu=torch.zeros(2, 3),
        quadrupole=torch.zeros(2, 3, 3),
        hfvr=hfvr,
        valence_width=torch.ones(2, 1),
        alpha=_alpha([1, 8], hfvr),
        damping=torch.ones(2, 1),
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("damped-cliff", 1.0),
        ("damped-amoeba", 2.0),
        ("undamped", 3.0),
    ],
)
def test_explicit_electrostatics_dispatch_ledgers_and_input_immutability(mode, expected):
    calls = []

    def elst_kernel(**kwargs):
        calls.append(mode)
        for name in (
            "qA", "qA_0", "muA", "quadA", "Ka",
            "qB", "qB_0", "muB", "quadB", "Kb",
        ):
            if name in kwargs:
                kwargs[name].add_(99.0)
        return torch.full((2,), expected)

    def induction_kernel(**kwargs):
        assert torch.equal(kwargs["qA"], torch.tensor([[0.4], [-0.4]]))
        for name in (
            "qA", "muA", "quadA", "hirshfeld_volume_ratio_A",
            "qB", "muB", "quadB", "hirshfeld_volume_ratio_B",
        ):
            kwargs[name].mul_(0.0)
        return torch.tensor([4.0, 5.0]), {
            "converged": True,
            "iterations": 7,
            "residual": 1.0e-9,
        }

    d3_calls = 0

    def dispersion_kernel(batch, params):
        nonlocal d3_calls
        d3_calls += 1
        return torch.tensor([6.0, 7.0])

    provider = LongRangeSAPTProvider(
        PhysicsConfig(electrostatics_mode=mode),
        electrostatics_kernels={mode: elst_kernel},
        induction_kernel=induction_kernel,
        dispersion_kernel=dispersion_kernel,
    )
    props_a = _props()
    props_b = _props()
    before_a = {
        name: getattr(props_a, name).clone() for name in props_a.__dataclass_fields__
    }
    before_b = {
        name: getattr(props_b, name).clone() for name in props_b.__dataclass_fields__
    }

    batch = _batch()
    if mode == "damped-amoeba":
        batch.amoeba_K_A = torch.ones(2)
        batch.amoeba_K_B = torch.ones(2)
    result = provider(batch, props_a, props_b)

    assert calls == [mode]
    assert d3_calls == 1
    for name, expected_tensor in before_a.items():
        assert torch.equal(getattr(props_a, name), expected_tensor), name
    for name, expected_tensor in before_b.items():
        assert torch.equal(getattr(props_b, name), expected_tensor), name
    assert result.pair_elst.tolist() == [expected, expected]
    assert result.dimer_elst.tolist() == [2 * expected]
    assert result.dimer_ind.tolist() == [9.0]
    assert result.dimer_disp.tolist() == [13.0]
    assert result.induction_diagnostics.converged
    assert result.induction_diagnostics.iterations == 7


def test_nonconvergence_policy_warns_or_raises():
    def elst_kernel(**kwargs):
        return torch.zeros(2)

    def induction_kernel(**kwargs):
        return torch.zeros(2), {
            "converged": False,
            "iterations": 2,
            "residual": 0.5,
        }

    def dispersion_kernel(batch, params):
        return torch.zeros(2)

    common = dict(
        electrostatics_kernels={"undamped": elst_kernel},
        induction_kernel=induction_kernel,
        dispersion_kernel=dispersion_kernel,
    )
    warning_provider = LongRangeSAPTProvider(
        PhysicsConfig(
            electrostatics_mode="undamped",
            scf_nonconvergence="warn",
        ),
        **common,
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        warning_provider(_batch(), _props(), _props())

    raising_provider = LongRangeSAPTProvider(
        PhysicsConfig(
            electrostatics_mode="undamped",
            scf_nonconvergence="raise",
        ),
        **common,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        raising_provider(_batch(), _props(), _props())


def test_negative_raw_hfvr_and_damping_match_positive_physical_values():
    captured = {}

    def elst_kernel(**kwargs):
        captured["Ka"] = kwargs["Ka"].clone()
        captured["Kb"] = kwargs["Kb"].clone()
        return kwargs["Ka"] + kwargs["Kb"]

    def induction_kernel(**kwargs):
        captured["hfvr_a"] = kwargs["hirshfeld_volume_ratio_A"].clone()
        captured["hfvr_b"] = kwargs["hirshfeld_volume_ratio_B"].clone()
        return (
            kwargs["hirshfeld_volume_ratio_A"]
            + kwargs["hirshfeld_volume_ratio_B"],
            {"converged": True, "iterations": 1, "residual": 0.0},
        )

    common = dict(
        electrostatics_kernels={"damped-cliff": elst_kernel},
        induction_kernel=induction_kernel,
        dispersion_kernel=lambda batch, params: torch.zeros(2),
    )
    positive = _props()
    negative = AtomicPropertyBundle(
        q=positive.q.clone(),
        mu=positive.mu.clone(),
        quadrupole=positive.quadrupole.clone(),
        hfvr=-positive.hfvr,
        valence_width=-positive.valence_width,
        alpha=positive.alpha,
        damping=-positive.damping,
    )
    provider = LongRangeSAPTProvider(PhysicsConfig(), **common)
    positive_result = provider(_batch(), positive, positive)
    negative_result = provider(_batch(), negative, negative)

    assert torch.equal(captured["Ka"], torch.ones(2))
    assert torch.equal(captured["Kb"], torch.ones(2))
    assert torch.equal(captured["hfvr_a"], torch.ones(2))
    assert torch.equal(captured["hfvr_b"], torch.ones(2))
    assert torch.equal(negative_result.pair_elst, positive_result.pair_elst)
    assert torch.equal(negative_result.pair_ind, positive_result.pair_ind)
    assert torch.isfinite(negative_result.dimer_elst).all()
    assert torch.isfinite(negative_result.dimer_ind).all()


def test_both_thole_controls_are_forwarded_to_induction_backend():
    captured = {}

    def induction_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2), {
            "converged": True,
            "iterations": 1,
            "residual": 0.0,
        }

    provider = LongRangeSAPTProvider(
        PhysicsConfig(thole_direct=0.21, thole_mutual=0.67),
        electrostatics_kernels={"damped-cliff": lambda **kwargs: torch.zeros(2)},
        induction_kernel=induction_kernel,
        dispersion_kernel=lambda batch, params: torch.zeros(2),
    )
    provider(_batch(), _props(), _props())
    assert captured["thole_damping_param_direct"] == 0.21
    assert captured["thole_damping_param_mutual"] == 0.67


def test_full_cartesian_edges_and_single_aggregation_for_two_unequal_dimers():
    first = dimer_fused_data(
        RA=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
        ZA=torch.tensor([1, 1]),
        TQA=0,
        RB=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0], [0.0, 0.0, 9.0]]),
        ZB=torch.tensor([1, 1, 1]),
        TQB=0,
        dimer_ind=0,
        r_cut=4.0,
        r_cut_im=2.0,
        check_validity=False,
    )
    second = dimer_fused_data(
        RA=torch.tensor([[1.0, 0.0, 0.0]]),
        ZA=torch.tensor([1]),
        TQA=0,
        RB=torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 4.0]]),
        ZB=torch.tensor([1, 1]),
        TQB=0,
        dimer_ind=1,
        r_cut=4.0,
        r_cut_im=2.0,
        check_validity=False,
    )
    assert first.e_ABsr_source.numel() and first.e_ABlr_source.numel()
    assert second.e_ABsr_source.numel() and second.e_ABlr_source.numel()
    batch = ap3_fused_collate_update_no_target([first, second])
    assert batch.e_ABfull_source.numel() == 2 * 3 + 1 * 2
    assert batch.dimer_ind_full.tolist() == [0] * 6 + [1] * 2

    first_pairs = set(
        zip(
            batch.e_ABfull_source[:6].tolist(),
            batch.e_ABfull_target[:6].tolist(),
        )
    )
    second_pairs = set(
        zip(
            (batch.e_ABfull_source[6:] - 2).tolist(),
            (batch.e_ABfull_target[6:] - 3).tolist(),
        )
    )
    assert first_pairs == {(a, b) for a in range(2) for b in range(3)}
    assert second_pairs == {(a, b) for a in range(1) for b in range(2)}

    pair_values = torch.arange(1.0, 9.0)
    hfvr_a = torch.ones(3, 1)
    props_a = AtomicPropertyBundle(
        q=torch.zeros(3, 1), mu=torch.zeros(3, 3),
        quadrupole=torch.zeros(3, 3, 3), hfvr=hfvr_a,
        valence_width=torch.ones(3, 1), alpha=_alpha([1, 1, 1], hfvr_a),
        damping=torch.ones(3, 1),
    )
    hfvr_b = torch.ones(5, 1)
    props_b = AtomicPropertyBundle(
        q=torch.zeros(5, 1), mu=torch.zeros(5, 3),
        quadrupole=torch.zeros(5, 3, 3), hfvr=hfvr_b,
        valence_width=torch.ones(5, 1), alpha=_alpha([1] * 5, hfvr_b),
        damping=torch.ones(5, 1),
    )
    provider = LongRangeSAPTProvider(
        PhysicsConfig(electrostatics_mode="undamped"),
        electrostatics_kernels={"undamped": lambda **kwargs: pair_values},
        induction_kernel=lambda **kwargs: (
            pair_values,
            {"converged": True, "iterations": 1, "residual": 0.0},
        ),
        dispersion_kernel=lambda batch, params: pair_values,
    )
    result = provider(batch, props_a, props_b)
    assert result.dimer_elst.tolist() == [21.0, 15.0]
    assert result.dimer_ind.tolist() == [21.0, 15.0]
    assert result.dimer_disp.tolist() == [21.0, 15.0]

    missing_pair = batch.clone()
    missing_pair.e_ABfull_source = missing_pair.e_ABfull_source[:-1]
    missing_pair.e_ABfull_target = missing_pair.e_ABfull_target[:-1]
    missing_pair.dimer_ind_full = missing_pair.dimer_ind_full[:-1]
    with pytest.raises(ValueError, match="every intermonomer Cartesian pair"):
        provider(missing_pair, props_a, props_b)


def _validated_water_reference_case():
    row = pd.read_pickle("tests/dataset_data/water_dimer_pes3.pkl").iloc[0]
    data = qcel_dimer_to_fused_data(
        row["qcel_molecule"], dimer_ind=0, r_cut=5.0, r_cut_im=8.0
    )
    batch = ap3_fused_collate_update_no_target([data])
    damping = torch.tensor(
        [2.05109221104216, 1.65393856475232, 1.65393856475232],
        dtype=torch.float32,
    ).reshape(-1, 1)

    def properties(fragment):
        def tensor(column):
            return torch.tensor(row[column], dtype=torch.float32)

        hfvr = tensor(f"vol_ratios_{fragment} pbe0/atz").reshape(-1, 1)
        numbers = batch.ZA if fragment == "A" else batch.ZB
        return AtomicPropertyBundle(
            q=tensor(f"q_{fragment} pbe0/atz").reshape(-1, 1),
            mu=tensor(f"mu_{fragment} pbe0/atz"),
            quadrupole=tensor(f"theta_{fragment} pbe0/atz"),
            hfvr=hfvr,
            valence_width=tensor(f"val_widths_{fragment} pbe0/atz").reshape(-1, 1),
            alpha=_alpha(numbers, hfvr),
            damping=damping.clone(),
        )

    return batch, properties("A"), properties("B")


@pytest.mark.parametrize("mode", ["damped-cliff", "undamped"])
def test_provenance_bearing_independent_numeric_references(mode):
    references = json.loads(
        Path("tests/dataset_data/mace_long_range_references.json").read_text()
    )
    provenance = references["provenance"]
    assert "tests/test_classical_components.py" in provenance["validation_basis"]
    import hashlib

    for path_key, hash_key in (
        ("source_fixture", "source_fixture_sha256"),
        ("amoeba_evidence", "amoeba_fixture_sha256"),
    ):
        if path_key == "source_fixture":
            source_path = Path(provenance[path_key])
        else:
            source_path = Path("tests/dataset_data/amoeba_water_dimer_ref.pkl")
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == provenance[hash_key]
    identity = provenance["generator_identity"].encode()
    assert hashlib.sha256(identity).hexdigest() == provenance[
        "generator_identity_sha256"
    ]
    batch, props_a, props_b = _validated_water_reference_case()
    result = LongRangeSAPTProvider(PhysicsConfig(electrostatics_mode=mode))(
        batch, props_a, props_b
    )
    expected = references[mode]
    tolerance = references["provenance"]["tolerance"]
    assert torch.allclose(
        result.dimer_elst,
        torch.tensor([expected["electrostatics"]]),
        **tolerance,
    )
    assert torch.allclose(
        result.dimer_ind,
        torch.tensor([expected["induction"]]),
        **tolerance,
    )
    assert torch.allclose(
        result.dimer_disp,
        torch.tensor([expected["d3"]]),
        **tolerance,
    )


def test_split_thole_controls_preserve_legacy_default_and_are_active():
    batch, props_a, props_b = _validated_water_reference_case()
    kwargs = dict(
        ZA=batch.ZA,
        RA=batch.RA,
        qA=props_a.q,
        muA=props_a.mu,
        quadA=props_a.quadrupole,
        ZB=batch.ZB,
        RB=batch.RB,
        qB=props_b.q,
        muB=props_b.mu,
        quadB=props_b.quadrupole,
        e_AB_source=batch.e_ABfull_source,
        e_AB_target=batch.e_ABfull_target,
        e_AA_source=batch.e_AA_source,
        e_AA_target=batch.e_AA_target,
        e_BB_source=batch.e_BB_source,
        e_BB_target=batch.e_BB_target,
        hirshfeld_volume_ratio_A=props_a.hfvr.reshape(-1),
        hirshfeld_volume_ratio_B=props_b.hfvr.reshape(-1),
    )
    legacy = induced_dipole_induction_optimized_no_correction(**kwargs)
    explicit_legacy = induced_dipole_induction_optimized_no_correction(
        **kwargs,
        thole_damping_param_direct=0.39,
        thole_damping_param_mutual=0.39,
    )
    changed_direct = induced_dipole_induction_optimized_no_correction(
        **kwargs,
        thole_damping_param_direct=0.20,
        thole_damping_param_mutual=0.39,
    )
    changed_mutual = induced_dipole_induction_optimized_no_correction(
        **kwargs,
        thole_damping_param_direct=0.39,
        thole_damping_param_mutual=0.70,
    )
    assert torch.equal(legacy, explicit_legacy)
    assert not torch.allclose(legacy, changed_direct)
    assert not torch.allclose(legacy, changed_mutual)


@pytest.mark.parametrize(
    "mode", ["damped-cliff", "damped-amoeba", "undamped"]
)
def test_real_low_level_backends_are_finite_for_every_explicit_mode(mode):
    data = dimer_fused_data(
        RA=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ZA=torch.tensor([1, 1]),
        TQA=0,
        RB=torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.74]]),
        ZB=torch.tensor([1, 1]),
        TQB=0,
        dimer_ind=0,
        r_cut=5.0,
        r_cut_im=8.0,
    )
    batch = ap3_fused_collate_update_no_target([data])
    if mode == "damped-amoeba":
        batch.amoeba_K_A = torch.ones(2)
        batch.amoeba_K_B = torch.ones(2)
    hfvr = torch.ones(2, 1)
    props = AtomicPropertyBundle(
        q=torch.ones(2, 1),
        mu=torch.zeros(2, 3),
        quadrupole=torch.zeros(2, 3, 3),
        hfvr=hfvr,
        valence_width=torch.ones(2, 1),
        alpha=_alpha([1, 1], hfvr),
        damping=torch.ones(2, 1),
    )

    result = LongRangeSAPTProvider(
        PhysicsConfig(electrostatics_mode=mode, scf_nonconvergence="raise")
    )(batch, props, props)

    assert result.pair_elst.shape == (4,)
    assert result.pair_ind.shape == (4,)
    assert result.pair_disp.shape == (4,)
    assert torch.isfinite(result.dimer_elst).all()
    assert torch.isfinite(result.dimer_ind).all()
    assert torch.isfinite(result.dimer_disp).all()


def test_real_classical_separation_scan_has_quantitative_asymptotic_decay():
    hfvr = torch.ones(1, 1)
    props = AtomicPropertyBundle(
        q=torch.tensor([[0.5]]),
        mu=torch.zeros(1, 3),
        quadrupole=torch.zeros(1, 3, 3),
        hfvr=hfvr,
        valence_width=torch.ones(1, 1),
        alpha=_alpha([1], hfvr),
        damping=torch.ones(1, 1),
    )
    values = []
    for separation in (6.0, 8.0, 10.0, 12.0):
        data = dimer_fused_data(
            RA=torch.zeros(1, 3), ZA=torch.tensor([1]), TQA=0,
            RB=torch.tensor([[separation, 0.0, 0.0]]),
            ZB=torch.tensor([1]), TQB=0, dimer_ind=0,
            r_cut=5.0, r_cut_im=8.0, check_validity=False,
        )
        batch = ap3_fused_collate_update_no_target([data])
        result = LongRangeSAPTProvider(
            PhysicsConfig(electrostatics_mode="undamped")
        )(batch, props, props)
        values.append(
            (result.dimer_elst.item(), result.dimer_ind.item(), result.dimer_disp.item())
        )
    magnitudes = torch.tensor(values).abs()
    assert torch.all(magnitudes[1:] < magnitudes[:-1])
    distances = torch.tensor([6.0, 8.0, 10.0, 12.0])
    scaled = torch.stack(
        (
            magnitudes[:, 0] * distances,
            magnitudes[:, 1] * distances.pow(4),
            magnitudes[:, 2] * distances.pow(6),
        ),
        dim=1,
    )
    assert torch.all(scaled.std(dim=0) / scaled.mean(dim=0) < 0.06)


def test_assembled_energy_and_first_derivative_are_continuous_at_real_cutoff():
    from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
        APNet3D3_AtomType_MPNN,
    )
    from apnet_pt.mace.schema import ClassicalEnergyBundle, InductionDiagnostics

    core = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    ).double()
    config = PhysicsConfig()
    assert core.r_cut_im == config.neural_cutoff == 8.0

    def total_energy(distance_value):
        distance = torch.tensor(distance_value, dtype=torch.float64, requires_grad=True)
        envelope = core.smooth_pair_energy_envelope(distance.reshape(1))
        residual = torch.zeros(1, 4, dtype=torch.float64)
        residual[0, 1] = envelope[0] / distance.pow(3)
        pair = distance.reshape(1) * 0.0
        classical = ClassicalEnergyBundle(
            pair_elst=pair, pair_ind=pair, pair_disp=pair,
            dimer_elst=(1.0 / distance).reshape(1),
            dimer_ind=(-0.1 / distance.pow(4)).reshape(1),
            dimer_disp=(-0.01 / distance.pow(6)).reshape(1),
            induction_diagnostics=InductionDiagnostics(True, 1, 0.0),
            physics_config_hash=config.physics_hash,
        )
        energy = assemble_sapt_components(residual, classical).sum()
        derivative = torch.autograd.grad(energy, distance)[0]
        return float(energy), float(derivative)

    left = total_energy(8.0 - 1.0e-4)
    center = total_energy(8.0)
    right = total_energy(8.0 + 1.0e-4)
    assert abs(left[0] - right[0]) < 5.0e-6
    assert abs(left[1] - right[1]) < 5.0e-6
    assert abs(center[1] - 0.5 * (left[1] + right[1])) < 5.0e-6


def test_sapt_assembly_adds_d3_once_and_honors_no_disp_nn():
    provider = LongRangeSAPTProvider(
        PhysicsConfig(electrostatics_mode="undamped"),
        electrostatics_kernels={"undamped": lambda **kwargs: torch.tensor([1.0, 2.0])},
        induction_kernel=lambda **kwargs: (
            torch.tensor([3.0, 4.0]),
            {"converged": True, "iterations": 1, "residual": 0.0},
        ),
        dispersion_kernel=lambda batch, params: torch.tensor([5.0, 6.0]),
    )
    classical = provider(_batch(), _props(), _props())
    residual = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    full = assemble_sapt_components(residual, classical)
    no_disp = assemble_sapt_components(residual, classical, no_disp_nn=True)

    assert full.tolist() == [[13.0, 20.0, 37.0, 51.0]]
    assert no_disp.tolist() == [[13.0, 20.0, 37.0, 11.0]]


def test_scf_convergence_norm_is_forwarded_to_induction_backend():
    # --scf_convergence_norm reaches the solver only by way of PhysicsConfig, so
    # this is the seam that decides whether the flag does anything at all.
    captured = {}

    def induction_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2), {
            "converged": True,
            "iterations": 1,
            "residual": 0.0,
        }

    def _run(config):
        captured.clear()
        provider = LongRangeSAPTProvider(
            config,
            electrostatics_kernels={"damped-cliff": lambda **kwargs: torch.zeros(2)},
            induction_kernel=induction_kernel,
            dispersion_kernel=lambda batch, params: torch.zeros(2),
        )
        provider(_batch(), _props(), _props())
        return captured["convergence_norm"]

    assert _run(PhysicsConfig()) == "l2"
    assert _run(PhysicsConfig(scf_convergence_norm="rms")) == "rms"
    assert _run(PhysicsConfig(scf_convergence_norm="max")) == "max"
