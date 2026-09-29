"""CLIFF2's Rackers/Thole induction as a MACE/AP3D3 backend.

The MACE route and the CLIFF2 pairwise route have always solved induction with
different kernels.  These tests cover the seam that lets a MACE experiment
select CLIFF2's functional instead: the adapters, the `PhysicsConfig`
selector, and the hash contract that keeps every checkpoint written before the
selector existed valid.
"""

from types import SimpleNamespace

import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    induced_dipole_induction_optimized_no_correction,
    rackers_thole_induction,
)
from apnet_pt.mace import long_range
from apnet_pt.mace.long_range import LongRangeSAPTProvider
from apnet_pt.mace.schema import (
    DEFAULT_INDUCTION_MODEL,
    INDUCTION_MODELS,
    AtomicPropertyBundle,
    PhysicsConfig,
)

# The hash `configs/mace-apnet/physics-v1.json` stamps and every v3 checkpoint
# on Phoenix carries.  Adding a physics field must not move it.
PHYSICS_V1_HASH = "bbaac8bc4d8f839ba73d822678fe61e63cf4abea267da0e57478a4879a02cbd0"


def _batch():
    """Two heteronuclear diatomics, far enough apart for the SCF to converge."""

    return SimpleNamespace(
        ZA=torch.tensor([1, 8]),
        RA=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        ZB=torch.tensor([1, 8]),
        RB=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=torch.float64),
        e_AA_source=torch.tensor([0, 1]),
        e_AA_target=torch.tensor([1, 0]),
        e_BB_source=torch.tensor([0, 1]),
        e_BB_target=torch.tensor([1, 0]),
        e_ABfull_source=torch.tensor([0, 0, 1, 1]),
        e_ABfull_target=torch.tensor([0, 1, 0, 1]),
        dimer_ind_full=torch.tensor([0, 0, 0, 0]),
        total_charge_A=torch.tensor([0.0]),
        total_charge_B=torch.tensor([0.0]),
    )


def _props():
    hfvr = torch.ones(2, 1, dtype=torch.float64)
    alpha = (
        constants.polarizability_table[torch.tensor([1, 8])]
        .reshape(-1, 1)
        .to(hfvr)
        * hfvr.abs().pow(4.0 / 3.0)
    )
    return AtomicPropertyBundle(
        q=torch.tensor([[0.4], [-0.4]], dtype=torch.float64),
        mu=torch.zeros(2, 3, dtype=torch.float64),
        quadrupole=torch.zeros(2, 3, 3, dtype=torch.float64),
        hfvr=hfvr,
        valence_width=torch.full((2, 1), 1.5, dtype=torch.float64),
        alpha=alpha,
        damping=torch.ones(2, 1, dtype=torch.float64),
    )


def _canonical_kwargs(batch, a, b, config):
    """Exactly what `LongRangeSAPTProvider._induction` sends a backend."""

    return dict(
        ZA=batch.ZA,
        RA=batch.RA,
        qA=a.q,
        muA=a.mu,
        quadA=a.quadrupole,
        ZB=batch.ZB,
        RB=batch.RB,
        qB=b.q,
        muB=b.mu,
        quadB=b.quadrupole,
        e_AB_source=batch.e_ABfull_source,
        e_AB_target=batch.e_ABfull_target,
        e_AA_source=batch.e_AA_source,
        e_AA_target=batch.e_AA_target,
        e_BB_source=batch.e_BB_source,
        e_BB_target=batch.e_BB_target,
        hirshfeld_volume_ratio_A=torch.abs(a.hfvr).reshape(-1),
        hirshfeld_volume_ratio_B=torch.abs(b.hfvr).reshape(-1),
        valence_widths_A=torch.abs(a.valence_width).reshape(-1),
        valence_widths_B=torch.abs(b.valence_width).reshape(-1),
        max_iterations=config.scf_max_iterations,
        convergence_threshold=config.scf_tolerance,
        thole_damping_param_direct=config.thole_direct,
        thole_damping_param_mutual=config.thole_mutual,
        convergence_norm=config.scf_convergence_norm,
    )


# --------------------------------------------------------------------------
# PhysicsConfig contract
# --------------------------------------------------------------------------


def test_the_default_is_the_kernel_the_route_already_used():
    assert DEFAULT_INDUCTION_MODEL == "ap3-no-correction"
    assert PhysicsConfig().induction_model == DEFAULT_INDUCTION_MODEL


@pytest.mark.parametrize("model", INDUCTION_MODELS)
def test_every_registered_model_is_accepted(model):
    assert PhysicsConfig(induction_model=model).induction_model == model


@pytest.mark.parametrize("bad", ["cliff2", "rackers", "", None, 3, "AP3-no-correction"])
def test_unregistered_models_are_refused(bad):
    with pytest.raises(ValueError, match="induction_model"):
        PhysicsConfig(induction_model=bad)


def test_the_default_leaves_the_published_physics_hash_alone():
    # Elision is the whole reason a new field is safe to add here: without it
    # every manifest and v3 checkpoint written before this field would fail
    # reconstruction against a hash it never chose.
    assert PhysicsConfig().physics_hash == PHYSICS_V1_HASH


def test_selecting_cliff2_changes_the_hash():
    # Elision must not extend to non-default values: a run that solves a
    # different induction functional is not the same physics.
    assert PhysicsConfig(induction_model="cliff2-rackers").physics_hash != (
        PHYSICS_V1_HASH
    )


def test_the_two_elided_defaults_are_independent():
    hashes = {
        PhysicsConfig().physics_hash,
        PhysicsConfig(scf_convergence_norm="rms").physics_hash,
        PhysicsConfig(induction_model="cliff2-rackers").physics_hash,
        PhysicsConfig(
            scf_convergence_norm="rms", induction_model="cliff2-rackers"
        ).physics_hash,
    }
    assert len(hashes) == 4


# --------------------------------------------------------------------------
# Backend registry and selection
# --------------------------------------------------------------------------


def test_every_registered_model_has_a_backend():
    _, induction_backends, _ = long_range._default_backends()
    assert set(induction_backends) == set(INDUCTION_MODELS)


@pytest.mark.parametrize("model", INDUCTION_MODELS)
def test_the_config_selects_the_backend(model):
    provider = LongRangeSAPTProvider(PhysicsConfig(induction_model=model))
    _, induction_backends, _ = long_range._default_backends()
    assert provider.induction_kernel is induction_backends[model]


def test_an_injected_kernel_outranks_the_selector():
    # Callers that inject are substituting the whole backend; honouring the
    # selector on top of that would silently ignore half of what they asked.
    def injected(**kwargs):
        return torch.zeros(2), {"converged": True, "iterations": 1, "residual": 0.0}

    provider = LongRangeSAPTProvider(
        PhysicsConfig(induction_model="cliff2-rackers"),
        induction_kernel=injected,
    )
    assert provider.induction_kernel is injected


# --------------------------------------------------------------------------
# Adapter fidelity
# --------------------------------------------------------------------------


def test_the_ap3_adapter_is_a_pure_pass_through():
    # The adapter exists only to absorb the valence widths.  If it perturbs the
    # historical kernel at all, every AP3-D3 checkpoint silently changes.
    batch, a, b = _batch(), _props(), _props()
    kwargs = _canonical_kwargs(batch, a, b, PhysicsConfig())

    adapted = long_range._ap3_no_correction_induction(**kwargs)
    direct = induced_dipole_induction_optimized_no_correction(
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"valence_widths_A", "valence_widths_B"}
        }
    )
    assert torch.equal(adapted, direct)


def test_the_cliff2_adapter_matches_a_direct_uniform_parameter_call():
    batch, a, b = _batch(), _props(), _props()
    config = PhysicsConfig(induction_model="cliff2-rackers")
    kwargs = _canonical_kwargs(batch, a, b, config)

    adapted = long_range._cliff2_rackers_induction(**kwargs)

    natom_a = a.natom
    natom_b = b.natom
    direct = rackers_thole_induction(
        ZA=batch.ZA,
        RA=batch.RA,
        qA=a.q,
        muA=a.mu,
        quadA=a.quadrupole,
        ZB=batch.ZB,
        RB=batch.RB,
        qB=b.q,
        muB=b.mu,
        quadB=b.quadrupole,
        e_AB_source=batch.e_ABfull_source,
        e_AB_target=batch.e_ABfull_target,
        e_AA_source=batch.e_AA_source,
        e_AA_target=batch.e_AA_target,
        e_BB_source=batch.e_BB_source,
        e_BB_target=batch.e_BB_target,
        hirshfeld_volume_ratio_A=kwargs["hirshfeld_volume_ratio_A"],
        hirshfeld_volume_ratio_B=kwargs["hirshfeld_volume_ratio_B"],
        valence_widths_A=kwargs["valence_widths_A"],
        valence_widths_B=kwargs["valence_widths_B"],
        thole_direct_A=torch.full((natom_a,), config.thole_direct, dtype=torch.float64),
        thole_direct_B=torch.full((natom_b,), config.thole_direct, dtype=torch.float64),
        thole_mutual_A=torch.full((natom_a,), config.thole_mutual, dtype=torch.float64),
        thole_mutual_B=torch.full((natom_b,), config.thole_mutual, dtype=torch.float64),
        ind_overlap_A=torch.zeros(natom_a, dtype=torch.float64),
        ind_overlap_B=torch.zeros(natom_b, dtype=torch.float64),
        include_overlap=False,
        max_iterations=config.scf_max_iterations,
        convergence_threshold=config.scf_tolerance,
        convergence_norm=config.scf_convergence_norm,
    )
    assert torch.equal(adapted, direct)


def test_the_cliff2_adapter_translates_the_diagnostics_keys():
    # The two kernels disagree on diagnostics names; `_normalize_diagnostics`
    # speaks only the MACE one, so an untranslated dict raises KeyError deep in
    # the provider rather than anywhere a reader would look.
    batch, a, b = _batch(), _props(), _props()
    config = PhysicsConfig(induction_model="cliff2-rackers")
    kwargs = _canonical_kwargs(batch, a, b, config)

    _, diagnostics = long_range._cliff2_rackers_induction(
        **kwargs, return_diagnostics=True
    )
    assert set(diagnostics) == {"converged", "iterations", "residual"}
    assert diagnostics["converged"] is True
    assert isinstance(diagnostics["iterations"], int)
    assert diagnostics["iterations"] >= 1
    assert diagnostics["residual"] < config.scf_tolerance


# --------------------------------------------------------------------------
# End to end through the provider
# --------------------------------------------------------------------------


def _dimer_induction(model):
    provider = LongRangeSAPTProvider(
        PhysicsConfig(induction_model=model),
        dispersion_kernel=lambda batch, params: torch.zeros(
            batch.e_ABfull_source.numel(), dtype=torch.float64
        ),
    )
    bundle = provider(_batch(), _props(), _props())
    return bundle


@pytest.mark.parametrize("model", INDUCTION_MODELS)
def test_both_backends_produce_a_finite_converged_induction(model):
    bundle = _dimer_induction(model)
    assert torch.isfinite(bundle.dimer_ind).all()
    assert bundle.induction_diagnostics.converged
    assert bundle.induction_diagnostics.iterations >= 1


def test_the_two_backends_are_actually_different_physics():
    # If these agreed there would be nothing to select between, and the hash
    # change above would be gratuitous.
    ap3 = _dimer_induction("ap3-no-correction").dimer_ind
    cliff2 = _dimer_induction("cliff2-rackers").dimer_ind
    assert not torch.allclose(ap3, cliff2)


def test_the_selector_reaches_the_bundle_hash():
    assert _dimer_induction("cliff2-rackers").physics_config_hash == (
        PhysicsConfig(induction_model="cliff2-rackers").physics_hash
    )
