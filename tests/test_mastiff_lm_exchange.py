"""MASTIFF-form exchange anisotropy: local frames and the `mastiff-lm` mode.

The physics being pinned down here is MASTIFF Eq. (3),

    A_i = A_iso * (1 + xi_i),
    xi_i = sum_{l>0,m} a_{i,lm} * C_lm(theta_i, phi_i),

with `C_lm` the Racah-normalized real spherical harmonics read in a body-fixed
frame attached to atom `i`.  Because MASTIFF's radial factor is exactly CLIFF's
`S_ij`, the whole content of the mode is `K_i -> K_i * (1 + xi_i)`, so these
tests are about the angular factor and nothing else.

Every symmetry below is a property the energy must have for reasons outside the
model: rotating the whole dimer cannot change its energy, a monatomic fragment
has no frame to be anisotropic in, and reflecting an achiral construction must
leave it alone.  A learned frame is free to be wrong; it is not free to break
these.
"""


import pytest
import torch

from apnet_pt import local_frame
from apnet_pt.AtomPairwiseModels.mtp_mtp import cliff_exchange

DTYPE = torch.float64


def _random_rotation(seed):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((3, 3), dtype=DTYPE, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _unit(vectors):
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# The harmonics themselves
# --------------------------------------------------------------------------


def test_racah_harmonics_have_the_racah_normalization():
    """<C_lm^2> over the sphere is 1/(2l+1); this is what fixes the prefactor.

    MASTIFF writes `sqrt(4*pi/(2l+1)) * Y_lm` rather than `Y_lm`, and the two
    differ by a factor that is constant within an `l` but not across them.  Get
    it wrong and `a_20` silently means something 2.24x larger than `a_10` does.
    """
    generator = torch.Generator().manual_seed(0)
    unit = _unit(torch.randn((200_000, 3), dtype=DTYPE, generator=generator))
    values = local_frame.racah_harmonics_l1l2(unit)
    mean_square = (values * values).mean(dim=0)
    expected = torch.tensor([1 / 3] * 3 + [1 / 5] * 5, dtype=DTYPE)
    assert torch.allclose(mean_square, expected, atol=5e-3)


def test_racah_harmonics_are_orthogonal_on_the_sphere():
    generator = torch.Generator().manual_seed(1)
    unit = _unit(torch.randn((200_000, 3), dtype=DTYPE, generator=generator))
    values = local_frame.racah_harmonics_l1l2(unit)
    gram = (values.unsqueeze(-1) * values.unsqueeze(-2)).mean(dim=0)
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    assert off_diagonal.abs().max().item() < 5e-3


def test_channel_indices_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        local_frame.channel_indices(("10", "33c"))


def test_channel_indices_rejects_duplicates():
    with pytest.raises(ValueError):
        local_frame.channel_indices(("10", "10"))


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------


def test_frame_is_always_a_proper_rotation():
    """Including for the degenerate inputs, which must not produce NaN.

    A degenerate atom gets a valid rotation and a false validity flag rather
    than a NaN frame, precisely so the flag has to be carried alongside; that
    contract is what `channel_mask` and the packed columns depend on.
    """
    generator = torch.Generator().manual_seed(2)
    v_z = torch.randn((64, 3), dtype=DTYPE, generator=generator)
    v_x = torch.randn((64, 3), dtype=DTYPE, generator=generator)
    v_z[0] = 0.0                       # no polar axis at all
    v_x[1] = 0.0                       # no azimuthal reference
    v_x[2] = v_z[2] * 2.5              # azimuthal reference collinear with e_z
    frame, z_valid, x_valid = local_frame.gram_schmidt_frame(v_z, v_x)
    assert torch.isfinite(frame).all()
    determinant = torch.det(frame)
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=1e-12)
    identity = torch.eye(3, dtype=DTYPE).expand_as(frame)
    assert torch.allclose(frame @ frame.transpose(-1, -2), identity, atol=1e-12)
    assert not bool(z_valid[0])
    assert not bool(x_valid[0])
    assert bool(z_valid[1]) and not bool(x_valid[1])
    assert bool(z_valid[2]) and not bool(x_valid[2])


def test_frame_rotates_with_its_inputs():
    generator = torch.Generator().manual_seed(3)
    v_z = torch.randn((32, 3), dtype=DTYPE, generator=generator)
    v_x = torch.randn((32, 3), dtype=DTYPE, generator=generator)
    rotation = _random_rotation(4)
    frame, _, _ = local_frame.gram_schmidt_frame(v_z, v_x)
    rotated, _, _ = local_frame.gram_schmidt_frame(
        v_z @ rotation.T, v_x @ rotation.T
    )
    assert torch.allclose(rotated, frame @ rotation.T, atol=1e-12)


# --------------------------------------------------------------------------
# xi, the angular factor
# --------------------------------------------------------------------------


def _xi_case(seed, labels=local_frame.PARITY_EVEN_LABELS):
    generator = torch.Generator().manual_seed(seed)
    active = local_frame.channel_indices(labels)
    coefficients = 0.2 * torch.randn(
        (16, len(active)), dtype=DTYPE, generator=generator
    )
    v_z = torch.randn((16, 3), dtype=DTYPE, generator=generator)
    v_x = torch.randn((16, 3), dtype=DTYPE, generator=generator)
    external = _unit(torch.randn((16, 3), dtype=DTYPE, generator=generator))
    return active, coefficients, v_z, v_x, external


def _xi(active, coefficients, v_z, v_x, external):
    frame, z_valid, x_valid = local_frame.gram_schmidt_frame(v_z, v_x)
    return local_frame.local_angular_xi(
        coefficients, frame, z_valid, x_valid, external, active
    )


def test_xi_is_rotation_invariant():
    active, coefficients, v_z, v_x, external = _xi_case(5)
    rotation = _random_rotation(6)
    expected = _xi(active, coefficients, v_z, v_x, external)
    got = _xi(
        active,
        coefficients,
        v_z @ rotation.T,
        v_x @ rotation.T,
        external @ rotation.T,
    )
    assert torch.allclose(got, expected, atol=1e-12)


def test_xi_is_mirror_invariant_on_the_parity_even_channels():
    """`e_y = e_z x e_x` is a pseudovector, so `sin(m phi)` flips under a
    reflection while `cos(m phi)` does not.  The default channel set is exactly
    the non-flipping one, which is what keeps an achiral dimer's exchange
    energy achiral."""
    active, coefficients, v_z, v_x, external = _xi_case(7)
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE))
    expected = _xi(active, coefficients, v_z, v_x, external)
    got = _xi(
        active, coefficients, v_z @ mirror, v_x @ mirror, external @ mirror
    )
    assert torch.allclose(got, expected, atol=1e-12)


def test_xi_is_not_mirror_invariant_once_the_sine_channels_are_admitted():
    """The complement of the test above: this is a real asymmetry, not a
    tolerance.  It is why `all` is opt-in and documented as chiral."""
    active, coefficients, v_z, v_x, external = _xi_case(
        8, local_frame.RACAH_L1L2_LABELS
    )
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE))
    expected = _xi(active, coefficients, v_z, v_x, external)
    got = _xi(
        active, coefficients, v_z @ mirror, v_x @ mirror, external @ mirror
    )
    assert (got - expected).abs().max().item() > 1e-2


def test_xi_vanishes_when_no_frame_exists():
    """A monatomic fragment has no neighbours, hence no polar axis, hence no
    anisotropy to express.  MASTIFF gets this from a symmetry table; here it
    falls out of the frame being invalid."""
    active = local_frame.channel_indices(local_frame.PARITY_EVEN_LABELS)
    coefficients = torch.full((1, len(active)), 0.5, dtype=DTYPE)
    frame, z_valid, x_valid = local_frame.gram_schmidt_frame(
        torch.zeros((1, 3), dtype=DTYPE), torch.zeros((1, 3), dtype=DTYPE)
    )
    external = torch.tensor([[0.3, -0.5, 0.81]], dtype=DTYPE)
    external = _unit(external)
    xi = local_frame.local_angular_xi(
        coefficients, frame, z_valid, x_valid, external, active
    )
    assert torch.equal(xi, torch.zeros_like(xi))


def test_xi_ignores_the_azimuth_when_the_environment_is_linear():
    """A C-infinity-v site has a polar axis but no azimuthal reference.  The
    `m != 0` channels are masked, so an arbitrary choice of `e_x` cannot leak
    into the energy -- this is MASTIFF's benzene-hydrogen reduction to
    {a_10, a_20}, obtained from geometry rather than declared."""
    active = local_frame.channel_indices(local_frame.PARITY_EVEN_LABELS)
    coefficients = torch.tensor([[0.4, 0.3, -0.2, 0.5, 0.1]], dtype=DTYPE)
    v_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    external = _unit(torch.tensor([[0.4, 0.2, 0.7]], dtype=DTYPE))
    values = []
    for reference in ([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.6, -0.8, 0.0]]):
        frame, z_valid, x_valid = local_frame.gram_schmidt_frame(
            v_z, torch.tensor(reference, dtype=DTYPE) * 0.0
        )
        values.append(
            local_frame.local_angular_xi(
                coefficients, frame, z_valid, x_valid, external, active
            )
        )
    for value in values[1:]:
        assert torch.equal(value, values[0])


# --------------------------------------------------------------------------
# The cuequivariance backend
# --------------------------------------------------------------------------


def test_cuequivariance_backend_matches_the_closed_form():
    cue = pytest.importorskip("cuequivariance_torch")
    del cue
    module = local_frame.CueRacahHarmonics().to(DTYPE)
    generator = torch.Generator().manual_seed(9)
    unit = _unit(torch.randn((512, 3), dtype=DTYPE, generator=generator))
    assert module.fit_residual < 1e-10
    assert torch.allclose(
        module(unit), local_frame.racah_harmonics_l1l2(unit), atol=1e-10
    )


def test_resolve_harmonics_auto_never_raises():
    harmonics = local_frame.resolve_harmonics("auto")
    unit = _unit(torch.tensor([[0.2, -0.3, 0.9]], dtype=DTYPE))
    assert harmonics(unit).shape == (1, 8)


def test_resolve_harmonics_torch_is_the_closed_form():
    assert local_frame.resolve_harmonics("torch") is (
        local_frame.racah_harmonics_l1l2
    )


def test_resolve_harmonics_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        local_frame.resolve_harmonics("numpy")


# --------------------------------------------------------------------------
# `cliff_exchange` in `mastiff-lm` mode
# --------------------------------------------------------------------------


def _exchange_case():
    return dict(
        RA=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, -0.2]], dtype=DTYPE),
        RB=torch.tensor([[2.6, -0.4, 0.7]], dtype=DTYPE),
        e_AB_source=torch.tensor([0, 1]),
        e_AB_target=torch.tensor([0, 0]),
        valence_widths_A=torch.tensor([0.4, 0.45], dtype=DTYPE),
        valence_widths_B=torch.tensor([0.5], dtype=DTYPE),
        K_exch_A=torch.tensor([2.0, 1.7], dtype=DTYPE),
        K_exch_B=torch.tensor([1.5], dtype=DTYPE),
    )


def _mastiff_kwargs(coefficient_scale=1.0, seed=10):
    generator = torch.Generator().manual_seed(seed)
    active = local_frame.channel_indices(local_frame.PARITY_EVEN_LABELS)
    frame_A, valid_z_A, valid_x_A = local_frame.gram_schmidt_frame(
        torch.tensor([[0.0, 0.3, 1.0], [1.0, 0.0, 0.1]], dtype=DTYPE),
        torch.tensor([[1.0, 0.2, 0.0], [0.0, 1.0, 0.4]], dtype=DTYPE),
    )
    frame_B, valid_z_B, valid_x_B = local_frame.gram_schmidt_frame(
        torch.tensor([[0.2, -1.0, 0.3]], dtype=DTYPE),
        torch.tensor([[1.0, 0.1, -0.5]], dtype=DTYPE),
    )
    return dict(
        anisotropy_mode="mastiff-lm",
        anisotropy_channels=active,
        anisotropy_A=coefficient_scale
        * 0.15
        * torch.randn((2, len(active)), dtype=DTYPE, generator=generator),
        anisotropy_B=coefficient_scale
        * 0.15
        * torch.randn((1, len(active)), dtype=DTYPE, generator=generator),
        frame_A=frame_A,
        frame_B=frame_B,
        frame_valid_A=torch.stack((valid_z_A, valid_x_A), dim=-1).to(DTYPE),
        frame_valid_B=torch.stack((valid_z_B, valid_x_B), dim=-1).to(DTYPE),
    )


def test_mastiff_zero_coefficients_are_the_exact_isotropic_limit():
    """Zero-init readouts must make enabling the mode a numerical no-op, so a
    warm-started arm reproduces its parent's benchmark to the printed digits
    before a single step is taken.  `torch.equal`, not `allclose`: `1 + 0` is
    exactly 1 and multiplying by it is exact in any dtype."""
    case = _exchange_case()
    kwargs = _mastiff_kwargs(coefficient_scale=0.0)
    assert torch.equal(cliff_exchange(**case, **kwargs), cliff_exchange(**case))


def test_mastiff_exchange_is_rotation_invariant():
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    rotation = _random_rotation(11)
    expected = cliff_exchange(**case, **kwargs)
    rotated = dict(case)
    rotated["RA"] = case["RA"] @ rotation.T
    rotated["RB"] = case["RB"] @ rotation.T
    turned = dict(kwargs)
    turned["frame_A"] = kwargs["frame_A"] @ rotation.T
    turned["frame_B"] = kwargs["frame_B"] @ rotation.T
    got = cliff_exchange(**rotated, **turned)
    assert torch.allclose(got, expected, atol=1e-12)


def test_mastiff_exchange_is_monomer_swap_symmetric():
    """`xi_i` reads `+rhat` and `xi_j` reads `-rhat`; exchanging the monomers
    has to exchange those too, or the pair energy depends on labelling."""
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    expected = cliff_exchange(**case, **kwargs)
    swapped = dict(
        RA=case["RB"],
        RB=case["RA"],
        e_AB_source=case["e_AB_target"],
        e_AB_target=case["e_AB_source"],
        valence_widths_A=case["valence_widths_B"],
        valence_widths_B=case["valence_widths_A"],
        K_exch_A=case["K_exch_B"],
        K_exch_B=case["K_exch_A"],
        anisotropy_mode="mastiff-lm",
        anisotropy_channels=kwargs["anisotropy_channels"],
        anisotropy_A=kwargs["anisotropy_B"],
        anisotropy_B=kwargs["anisotropy_A"],
        frame_A=kwargs["frame_B"],
        frame_B=kwargs["frame_A"],
        frame_valid_A=kwargs["frame_valid_B"],
        frame_valid_B=kwargs["frame_valid_A"],
    )
    assert torch.allclose(cliff_exchange(**swapped), expected, atol=1e-12)


def test_mastiff_matches_a_hand_written_reference():
    """Independent evaluation of `K_i K_j S_ij (1 + xi_i)(1 + xi_j)`.

    The symmetry tests above all pass for a model that computed the wrong
    energy in a consistent way; this one does not.
    """
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    isotropic = cliff_exchange(**case)
    delta = case["RB"][case["e_AB_target"]] - case["RA"][case["e_AB_source"]]
    rhat = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    active = list(kwargs["anisotropy_channels"])
    xi_i = local_frame.local_angular_xi(
        kwargs["anisotropy_A"][case["e_AB_source"]],
        kwargs["frame_A"][case["e_AB_source"]],
        kwargs["frame_valid_A"][case["e_AB_source"]][:, 0] > 0.5,
        kwargs["frame_valid_A"][case["e_AB_source"]][:, 1] > 0.5,
        rhat,
        active,
    )
    xi_j = local_frame.local_angular_xi(
        kwargs["anisotropy_B"][case["e_AB_target"]],
        kwargs["frame_B"][case["e_AB_target"]],
        kwargs["frame_valid_B"][case["e_AB_target"]][:, 0] > 0.5,
        kwargs["frame_valid_B"][case["e_AB_target"]][:, 1] > 0.5,
        -rhat,
        active,
    )
    expected = isotropic * (1.0 + xi_i) * (1.0 + xi_j)
    assert torch.allclose(cliff_exchange(**case, **kwargs), expected, atol=1e-14)
    # And the angular factor actually did something.
    assert (cliff_exchange(**case, **kwargs) - isotropic).abs().max() > 1e-6


def test_mastiff_without_a_frame_fails_closed():
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    kwargs.pop("frame_A")
    with pytest.raises(ValueError, match="frame_A"):
        cliff_exchange(**case, **kwargs)


def test_mastiff_without_channels_fails_closed():
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    kwargs["anisotropy_channels"] = ()
    with pytest.raises(ValueError, match="anisotropy_channels"):
        cliff_exchange(**case, **kwargs)


def test_mastiff_accepts_the_cuequivariance_harmonics():
    cue = pytest.importorskip("cuequivariance_torch")
    del cue
    case = _exchange_case()
    kwargs = _mastiff_kwargs()
    expected = cliff_exchange(**case, **kwargs)
    got = cliff_exchange(
        **case, **kwargs, harmonics=local_frame.CueRacahHarmonics().to(DTYPE)
    )
    assert torch.allclose(got, expected, atol=1e-10)


# --------------------------------------------------------------------------
# The learned frame module
# --------------------------------------------------------------------------


def _frame_module(seed=12):
    torch.manual_seed(seed)
    return local_frame.LearnedLocalFrame(
        n_embed=4, n_message=2, n_rbf=6, n_neuron=16, r_cut=5.0
    ).to(DTYPE)


def _frame_inputs(positions):
    n_atoms = positions.size(0)
    source, target = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                source.append(i)
                target.append(j)
    e_source = torch.tensor(source, dtype=torch.long)
    e_target = torch.tensor(target, dtype=torch.long)
    delta = positions.index_select(0, e_target) - positions.index_select(
        0, e_source
    )
    distance = torch.linalg.vector_norm(delta, dim=-1)
    return e_source, e_target, delta / distance.unsqueeze(-1), distance


def test_learned_frame_is_equivariant():
    """The MLP sees only rotation-invariant features -- hidden states and
    distances -- and its outputs are weights on the *edge unit vectors*, so the
    frame it builds turns with the molecule rather than with the lab."""
    module = _frame_module()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.3], [-0.4, 1.3, 0.5], [0.6, -0.9, 1.4]],
        dtype=DTYPE,
    )
    h_list = torch.randn((4, 3, 4), dtype=DTYPE)
    e_source, e_target, unit, distance = _frame_inputs(positions)
    frame, z_valid, x_valid = module(
        h_list, unit, distance, e_source, e_target, 4
    )
    rotation = _random_rotation(13)
    _, _, unit_r, distance_r = _frame_inputs(positions @ rotation.T)
    frame_r, z_valid_r, x_valid_r = module(
        h_list, unit_r, distance_r, e_source, e_target, 4
    )
    assert torch.allclose(frame_r, frame @ rotation.T, atol=1e-11)
    assert torch.equal(z_valid_r, z_valid)
    assert torch.equal(x_valid_r, x_valid)


def test_learned_frame_reports_a_monatomic_fragment_as_invalid():
    module = _frame_module()
    h_list = torch.randn((1, 3, 4), dtype=DTYPE)
    empty_long = torch.zeros((0,), dtype=torch.long)
    frame, z_valid, x_valid = module(
        h_list,
        torch.zeros((0, 3), dtype=DTYPE),
        torch.zeros((0,), dtype=DTYPE),
        empty_long,
        empty_long,
        1,
    )
    assert not bool(z_valid[0])
    assert not bool(x_valid[0])
    assert torch.isfinite(frame).all()


def test_learned_frame_gradients_reach_the_mlp():
    module = _frame_module()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.3], [-0.4, 1.3, 0.5]], dtype=DTYPE
    )
    h_list = torch.randn((3, 3, 4), dtype=DTYPE)
    e_source, e_target, unit, distance = _frame_inputs(positions)
    frame, _, _ = module(h_list, unit, distance, e_source, e_target, 3)
    frame.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


def test_learned_frame_is_smooth_at_the_cutoff():
    """An edge crossing `r_cut` must not step the frame, or the energy is
    discontinuous in the geometry and the forces are wrong there."""
    module = _frame_module()
    r_cut = module.r_cut
    base = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, -0.3]], dtype=DTYPE)
    e_source = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    e_target = torch.tensor([1, 0, 2, 0], dtype=torch.long)
    frames = []
    for offset in (-1e-6, 1e-6):
        far = torch.tensor([[0.0, 0.0, r_cut + offset]], dtype=DTYPE)
        positions = torch.cat((base, far), dim=0)
        delta = positions.index_select(0, e_target) - positions.index_select(
            0, e_source
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        unit = delta / distance.unsqueeze(-1)
        h_list = torch.zeros((3, 3, 4), dtype=DTYPE)
        frame, _, _ = module(h_list, unit, distance, e_source, e_target, 3)
        frames.append(frame)
    assert torch.allclose(frames[0], frames[1], atol=1e-5)
