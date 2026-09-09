"""Tests of vectorized symmetry-equivalent body-frame averaging."""

import pytest
import torch

from apnet_pt.mastiff_frames import AveragedFrameHarmonics


def setup():
    p = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.6], [-1.0, 0.0, 0.6]], dtype=torch.float64
    )
    targets = torch.tensor([[2.0, 1.0, 3.0], [-2.0, 0.3, 2.0]], dtype=torch.float64)
    full = torch.tensor([[0, 1, 2], [0, 2, 1]])
    axial = torch.tensor([[1, 0], [2, 0]])
    return p, targets, full, axial


def test_rigid_motion_reflection_and_permutation():
    p, targets, full, axial = setup()
    module = AveragedFrameHarmonics()
    d = targets[None] - p[:, None]
    expected = module(p, d, full, axial)
    for q in (
        torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=p.dtype
        ),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=p.dtype)),
    ):
        torch.testing.assert_close(module(p @ q.T + 10, d @ q.T, full, axial), expected)
    permutation = torch.tensor([2, 0, 1])
    inverse = permutation.argsort()
    actual = module(p[permutation], d[permutation], inverse[full], inverse[axial])
    torch.testing.assert_close(actual, expected[permutation])
    torch.testing.assert_close(module(p, d, full.flip(0), axial.flip(0)), expected)
    assert torch.equal(
        expected[..., [2, 5, 7]], torch.zeros_like(expected[..., [2, 5, 7]])
    )


def test_average_gradients_and_invalid_plans():
    p, targets, full, axial = setup()
    p.requires_grad_()
    module = AveragedFrameHarmonics()
    assert torch.autograd.gradcheck(
        lambda x: module(x, targets[None] - x[:, None], full, axial), (p,)
    )
    with pytest.raises(ValueError, match="both"):
        module(p, targets[None] - p[:, None], full, torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="degenerate axial"):
        module(p, targets[None] - p[:, None], full[:0], torch.tensor([[1, 1]]))
    with pytest.raises(ValueError, match="index outside"):
        module(p, targets[None] - p[:, None], torch.tensor([[-1, 1, 2]]), axial[:0])
    actual = module(p, targets[None] - p[:, None], full[:0], axial[:0])
    assert torch.equal(actual, torch.zeros_like(actual))


def test_cue_average_parity():
    pytest.importorskip("cuequivariance_torch")
    p, targets, full, axial = setup()
    d = targets[None] - p[:, None]
    torch.testing.assert_close(
        AveragedFrameHarmonics("cuequivariance", "naive")(p, d, full, axial),
        AveragedFrameHarmonics()(p, d, full, axial),
    )
