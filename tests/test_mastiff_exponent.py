"""Energy-kernel contracts, independent of the atomic feature producer."""

import math

import torch


def test_exponent_anisotropy_scales_the_entire_overlap_argument():
    from apnet_pt.mastiff_exponent import slater_exponent_exchange

    # A_i A_j=6, B_i=B_j=1, r=1 and exp((s_i+s_j)/2)=2:
    # x=2, E=6*(1+2+4/3)*exp(-2)=26*exp(-2).
    values = [2.0, 3.0, 1.0, 1.0, 1.0, math.log(4), 0.0]
    inputs = [torch.tensor([v], dtype=torch.float64) for v in values]
    actual = slater_exponent_exchange(*inputs)
    torch.testing.assert_close(
        actual,
        torch.tensor([26 * math.exp(-2)], dtype=torch.float64),
        rtol=1e-14,
        atol=0,
    )


def test_tiny_float32_decay_has_finite_log_parameter_gradients():
    from apnet_pt.mastiff_exponent import slater_exponent_exchange

    log_b = torch.tensor([math.log(1e-25)], requires_grad=True)
    one = torch.ones(1)
    zero = torch.zeros(1)
    energy = slater_exponent_exchange(
        one, one, log_b.exp(), log_b.exp(), one, zero, zero
    )
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(log_b.grad).all()
