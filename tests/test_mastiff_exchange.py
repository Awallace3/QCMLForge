"""MASTIFF equations and CANplugin conventions, independent of fitted weights."""
import math

import pytest
import torch

from apnet_pt.mastiff import MastiffExchange, RacahHarmonics, local_frames

DTYPE = torch.float64


def test_harmonics_against_trigonometric_definition():
    torch.manual_seed(5)
    v = torch.randn(32, 3, dtype=DTYPE)
    v = v / v.norm(dim=-1, keepdim=True)
    x, y, z = v.unbind(-1)
    phi = torch.atan2(y, x)
    s = torch.sqrt(1 - z * z)
    expected = torch.stack((z, s * phi.cos(), s * phi.sin(),
                            (3 * z * z - 1) / 2,
                            math.sqrt(3) * z * s * phi.cos(),
                            math.sqrt(3) * z * s * phi.sin(),
                            math.sqrt(3) / 2 * s.square() * (2 * phi).cos(),
                            math.sqrt(3) / 2 * s.square() * (2 * phi).sin()), -1)
    torch.testing.assert_close(RacahHarmonics()(v), expected)


def test_cue_matches_reference_values_and_gradients():
    pytest.importorskip('cuequivariance_torch')
    torch.manual_seed(6)
    v = torch.randn(20, 3, dtype=DTYPE, requires_grad=True)
    ref = RacahHarmonics()(v)
    actual = RacahHarmonics(backend='cuequivariance', method='naive')(v)
    torch.testing.assert_close(actual, ref)
    weights = torch.randn_like(ref)
    g1 = torch.autograd.grad((ref * weights).sum(), v, retain_graph=True)[0]
    g2 = torch.autograd.grad((actual * weights).sum(), v)[0]
    torch.testing.assert_close(g1, g2)


def sample():
    r = torch.tensor([[1., 2., 3.], [-2., 1., .5]], dtype=DTYPE)
    a = torch.tensor([2., 3.], dtype=DTYPE)
    b = torch.tensor([1.2, 1.7], dtype=DTYPE)
    c = torch.arange(16, dtype=DTYPE).reshape(2, 8) / 100
    f = torch.eye(3, dtype=DTYPE).expand(2, 3, 3)
    return r, a, b, c, f


def test_exact_isotropic_limit_and_unclipped_polynomial():
    r, a, b, c, f = sample()
    model = MastiffExchange()
    zero = torch.zeros_like(c)
    out = model(r, a, a.flip(0), b, b.flip(0), zero, zero, f, f)
    x = torch.sqrt(b * b.flip(0)) * r.norm(dim=-1)
    expected = a * a.flip(0) * (1 + x + x.square() / 3) * torch.exp(-x)
    torch.testing.assert_close(out, expected)
    # The literal paper form is not exp/tanh/clamped: it can be negative.
    c = zero.clone()
    c[:, 0] = -10
    out = model(r, a, a, b, b, c, zero, f, f)
    assert torch.all(out < 0)


def test_swap_rotation_translation_and_gradcheck():
    r, a, b, c, f = sample()
    model = MastiffExchange()
    out = model(r, a, a.flip(0), b, b.flip(0), c, c.flip(0), f, f)
    swapped = model(-r, a.flip(0), a, b.flip(0), b, c.flip(0), c, f, f)
    torch.testing.assert_close(swapped, out)
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 4.], [3., -1., 2.],
                                        [-2., 1., 1.]], dtype=DTYPE))
    rotated = model(r @ q.T, a, a.flip(0), b, b.flip(0), c, c.flip(0),
                    q @ f, q @ f)
    torch.testing.assert_close(rotated, out)
    inputs = tuple(t.clone().requires_grad_() for t in (r, a, a, b, b, c, c, f, f))
    assert torch.autograd.gradcheck(model, inputs)


def test_frames_zthenx_and_bisector():
    z = torch.tensor([[0., 0., 2.]], dtype=DTYPE)
    x = torch.tensor([[1., 0., 1.]], dtype=DTYPE)
    f = local_frames(z, x, kind='z-then-x')
    torch.testing.assert_close(f, torch.eye(3, dtype=DTYPE).unsqueeze(0))
    z = torch.tensor([[1., 0., 1.]], dtype=DTYPE)
    x = torch.tensor([[-1., 0., 1.]], dtype=DTYPE)
    f = local_frames(z, x, kind='bisector')
    torch.testing.assert_close(f.transpose(-1, -2) @ f,
                               torch.eye(3, dtype=DTYPE).unsqueeze(0))
    torch.testing.assert_close(f[..., 2], torch.tensor([[0., 0., 1.]], dtype=DTYPE))
    with pytest.raises(ValueError, match='degenerate'):
        local_frames(z, z, kind='z-then-x')


@pytest.mark.parametrize('kind', ['z-bisect', 'threefold'])
def test_zero_reference_bonds_fail_validation(kind):
    z = torch.tensor([[0., 0., 1.]], dtype=DTYPE)
    zero = torch.zeros_like(z)
    y = torch.tensor([[1., 0., 0.]], dtype=DTYPE)
    with pytest.raises(ValueError, match='degenerate'):
        local_frames(z, zero, y, kind=kind)


@pytest.mark.parametrize('kind', ['z-only', 'threefold'])
def test_axial_gauge_rotation_permutation_and_gradients(kind):
    torch.manual_seed(73)
    refs = torch.tensor([[.2, .3, 2.], [1., -.1, .4],
                         [-.4, .5, .3]], dtype=DTYPE, requires_grad=True)
    r = torch.tensor([[1., 2., 3.]], dtype=DTYPE)
    c = torch.arange(8, dtype=DTYPE).reshape(1, 8) / 10
    a = torch.ones(1, dtype=DTYPE)
    model = MastiffExchange(symmetry_i='axial', symmetry_j='axial')

    def energy(ref, displacement=r):
        args = (ref[0:1],) if kind == 'z-only' else tuple(ref[k:k+1] for k in range(3))
        f = local_frames(*args, kind=kind)
        return model(displacement, a, a, a, a, c, c, f, f).sum()

    assert torch.autograd.gradcheck(energy, (refs,))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
    rotated = (refs.detach() @ q.T).requires_grad_()
    torch.testing.assert_close(energy(rotated, r @ q.T), energy(refs))
    g = torch.autograd.grad(energy(refs), refs)[0]
    gr = torch.autograd.grad(energy(rotated, r @ q.T), rotated)[0]
    torch.testing.assert_close(gr, g @ q.T)
    if kind == 'threefold':
        torch.testing.assert_close(energy(refs[[2, 0, 1]]), energy(refs))
    else:
        # At x=y, the arbitrary transverse axis changes; axial energy is smooth.
        tied = torch.tensor([[.2, .2, 1.]], dtype=DTYPE, requires_grad=True)
        assert torch.autograd.gradcheck(lambda v: energy(v), (tied,))


@pytest.mark.parametrize('backend', ['torch', 'cuequivariance'])
def test_backend_poles_empty_and_fullgraph(backend):
    if backend == 'cuequivariance':
        pytest.importorskip('cuequivariance_torch')
    h = RacahHarmonics(backend=backend, method='naive')
    v = torch.tensor([[0., 0., 1.], [0., 0., -1.]], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(h, (v,))
    assert h(v[:0]).shape == (0, 8)
    torch.testing.assert_close(torch.compile(h, backend='eager', fullgraph=True)(v), h(v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_cue_cuda_float32_values_and_gradients():
    pytest.importorskip('cuequivariance_torch')
    torch.manual_seed(29)
    v = torch.randn(32, 3, device='cuda', requires_grad=True)
    ref = RacahHarmonics().cuda()(v)
    actual = RacahHarmonics(backend='cuequivariance').cuda()(v)
    torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-5)
    w = torch.randn_like(ref)
    g1 = torch.autograd.grad((ref*w).sum(), v, retain_graph=True)[0]
    g2 = torch.autograd.grad((actual*w).sum(), v)[0]
    torch.testing.assert_close(g1, g2, atol=5e-6, rtol=5e-5)


def test_can_raw_bond_frame_conventions():
    z = torch.tensor([[0., 0., 2.]], dtype=DTYPE)
    second = torch.tensor([[1., 0., 0.]], dtype=DTYPE)
    f = local_frames(z, second, kind='bisector')
    torch.testing.assert_close(f[..., 2], (z + second) / (z + second).norm())
    torch.testing.assert_close(f[..., 0], torch.tensor([[0., 1., 0.]], dtype=DTYPE))
    x = torch.tensor([[2., 0., 0.]], dtype=DTYPE)
    y = torch.tensor([[0., 1., 0.]], dtype=DTYPE)
    f = local_frames(z, x, y, kind='z-bisect')
    torch.testing.assert_close(f[..., 0], (x + y) / (x + y).norm())
    torch.testing.assert_close(f, local_frames(z, y, x, kind='z-bisect'))
    with pytest.raises(ValueError, match='degenerate'):
        local_frames(x, y, -x-y, kind='threefold')


def test_compile_fullgraph_and_empty_edges():
    r, a, b, c, f = sample()
    model = MastiffExchange()
    args = (r, a, a, b, b, c, c, f, f)
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(*args), model(*args))
    assert model(*(x[:0] for x in args)).shape == (0,)


def test_frame_coordinate_gradients_and_rigid_motion():
    torch.manual_seed(13)
    positions = torch.randn(6, 3, dtype=DTYPE, requires_grad=True)
    a = torch.ones(1, dtype=DTYPE)
    c = torch.arange(8, dtype=DTYPE).reshape(1, 8) / 20
    model = MastiffExchange()

    def energy(p):
        fi = local_frames((p[2]-p[0])[None], (p[3]-p[0])[None])
        fj = local_frames((p[4]-p[1])[None], (p[5]-p[1])[None])
        return model((p[1]-p[0])[None], a, a, a, a, c, c, fi, fj).sum()

    assert torch.autograd.gradcheck(energy, (positions,))
    forces = -torch.autograd.grad(energy(positions), positions)[0]
    torch.testing.assert_close(forces.sum(0), torch.zeros(3, dtype=DTYPE), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.linalg.cross(positions, forces).sum(0),
                               torch.zeros(3, dtype=DTYPE), atol=1e-12, rtol=0)
    torch.testing.assert_close(energy(positions + 9), energy(positions))


def test_bisector_reference_swap_with_c2v_and_signed_sines():
    z = torch.tensor([[0., 0., 2.]], dtype=DTYPE)
    x = torch.tensor([[1., 0., 0.]], dtype=DTYPE)
    f1, f2 = local_frames(z, x, kind='bisector'), local_frames(x, z, kind='bisector')
    r, a, b, c, f = sample()
    model = MastiffExchange(symmetry_i='c2v', symmetry_j='c2v')
    torch.testing.assert_close(model(r, a, a, b, b, c, c, f1, f),
                               model(r, a, a, b, b, c, c, f2, f))
    v = torch.tensor([[1., 1., 1.], [1., -1., 1.]], dtype=DTYPE)
    h = RacahHarmonics()(v)
    torch.testing.assert_close(h[0, [2, 5, 7]], -h[1, [2, 5, 7]])


def test_axial_mask_and_pole_gradients():
    r, a, b, c, f = sample()
    model = MastiffExchange(symmetry_i='axial', symmetry_j='isotropic')
    out = model(r, a, a, b, b, c, c, f, f)
    masked = torch.zeros_like(c)
    masked[:, [0, 3]] = c[:, [0, 3]]
    expected = MastiffExchange()(r, a, a, b, b, masked, torch.zeros_like(c), f, f)
    torch.testing.assert_close(out, expected)
    v = torch.tensor([[0., 0., 1.], [0., 0., -1.]], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(RacahHarmonics(), (v,))


def test_c2v_matches_canplugin_benzene_expression():
    # CAN example XML parameters: units nm, sqrt(kJ/mol), nm^-1.
    r = torch.tensor([[.23, .17, .31]], dtype=DTYPE)
    a = torch.tensor([86.22564], dtype=DTYPE)
    b = torch.tensor([36.96344], dtype=DTYPE)
    c = torch.zeros(1, 8, dtype=DTYPE)
    c[:, [0, 3, 6]] = torch.tensor([.8294600, .007720117, -.8290971], dtype=DTYPE)
    f = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    model = MastiffExchange(symmetry_i='c2v', symmetry_j='c2v')
    u = r / r.norm(dim=-1, keepdim=True)
    x, y, z = u.unbind(-1)
    even = c[:, 3] * (3*z*z-1)/2 + c[:, 6] * math.sqrt(.75)*(x*x-y*y)
    angular = (1+even+c[:, 0]*z)*(1+even-c[:, 0]*z)
    br = b*r.norm(dim=-1)
    expected = a*a*angular*(br*br/3+br+1)*torch.exp(-br)
    torch.testing.assert_close(model(r, a, a, b, b, c, c, f, f), expected)
