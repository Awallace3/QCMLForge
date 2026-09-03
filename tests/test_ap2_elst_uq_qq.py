"""The dipole-quadrupole and quadrupole-quadrupole electrostatics terms.

The published TensorFlow AP-Net2 omits both.  Its predecessor did not: the
``master`` tip of ``github.com/zachglick/apnet`` (commit ``e09955b``, which is
``593d655^``) summed qq, qu, qQ, uu, uQ and QQ in
``apnet/multipoles.py::eval_interaction``, and the ``sparse`` rewrite that
replaced it with ``KerasPairModel.mtp_elst`` dropped the last two while adding
a 3/2 factor on both quadrupole tensors.

``reference_T_cart`` below is that pre-rewrite ``T_cart`` transcribed verbatim,
so these tests pin ``_elst_uQ_QQ`` to the code it is meant to restore rather
than to a re-derivation of the multipole expansion.
"""

import numpy as np
import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN

MODEL_FACTORIES = [
    lambda **kwargs: APNet2_MPNN(**kwargs),
    lambda **kwargs: APNet2_AM_MPNN(atom_model=AtomMPNN(), **kwargs),
]


def reference_T_cart(RA, RB):
    """``apnet/multipoles.py::T_cart`` at ``593d655^``, transcribed."""
    dR = RB - RA
    R = np.linalg.norm(dR)
    delta = np.identity(3)

    T2 = (R ** -5) * (3 * np.outer(dR, dR) - R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (R ** -7) * -1.0 * (
        15 * np.multiply.outer(np.outer(dR, dR), dR)
        - 3 * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
    )

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R ** -9) * (
        105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15 * R * R * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + 3 * (R ** 4) * (
            dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1)
        )
    )
    return T2, T3, T4


def reference_uQ_QQ(dR_xyz, muA, muB, quadA, quadB):
    """``E_uQ + E_QQ`` from the pre-rewrite ``eval_interaction``, per edge."""
    out = np.empty(len(dR_xyz))
    origin = np.zeros(3)
    for edge in range(len(dR_xyz)):
        _, T3, T4 = reference_T_cart(origin, dR_xyz[edge])
        E_uQ = np.sum(
            T3
            * (
                np.multiply.outer(muA[edge], quadB[edge])
                - np.multiply.outer(muB[edge], quadA[edge])
            )
        ) * (-1.0 / 3.0)
        E_QQ = np.sum(
            T4 * np.multiply.outer(quadA[edge], quadB[edge])
        ) * (1.0 / 9.0)
        out[edge] = E_uQ + E_QQ
    return out


def random_multipoles(n_edge, seed):
    """Traceless symmetric quadrupoles, as both atom models are trained to emit."""
    rng = np.random.default_rng(seed)
    quads = []
    for _ in range(n_edge):
        raw = rng.normal(size=(3, 3))
        raw = 0.5 * (raw + raw.T)
        quads.append(raw - np.identity(3) * np.trace(raw) / 3.0)
    return (
        rng.normal(size=(n_edge, 3)),
        rng.normal(size=(n_edge, 3)),
        np.stack(quads),
    )


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_uQ_QQ_reproduces_pre_rewrite_kernel(model_factory):
    """``_elst_uQ_QQ`` must match the original numpy to float64 precision."""
    n_edge = 6
    muA, muB, quadA = random_multipoles(n_edge, seed=11)
    _, _, quadB = random_multipoles(n_edge, seed=12)
    rng = np.random.default_rng(13)
    # Keep well away from r=0; the terms fall off as r^-7 and r^-9.
    dR_xyz = rng.normal(size=(n_edge, 3)) + 4.0

    model = model_factory(elst_include_uQ_QQ=True)
    dR_xyz_t = torch.tensor(dR_xyz, dtype=torch.float64)
    dR_t = torch.sqrt((dR_xyz_t * dR_xyz_t).sum(-1))
    got = model._elst_uQ_QQ(
        dR_t,
        dR_xyz_t,
        1.0 / dR_t,
        torch.eye(3, dtype=torch.float64),
        torch.tensor(muA, dtype=torch.float64),
        torch.tensor(muB, dtype=torch.float64),
        torch.tensor(quadA, dtype=torch.float64),
        torch.tensor(quadB, dtype=torch.float64),
    )
    expected = reference_uQ_QQ(dR_xyz, muA, muB, quadA, quadB)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_flag_off_is_the_published_tensorflow_kernel(model_factory):
    """The default must leave ``mtp_elst`` bit-identical to before the flag."""
    n_edge = 4
    muA, muB, quadA = random_multipoles(n_edge, seed=21)
    _, _, quadB = random_multipoles(n_edge, seed=22)
    rng = np.random.default_rng(23)
    dR_xyz = rng.normal(size=(n_edge, 3)) + 4.0
    dR = np.linalg.norm(dR_xyz, axis=-1)

    inputs = {
        "qA": torch.tensor(rng.normal(size=(n_edge, 1)), dtype=torch.float32),
        "muA": torch.tensor(muA, dtype=torch.float32),
        "quadA": torch.tensor(quadA, dtype=torch.float32),
        "qB": torch.tensor(rng.normal(size=(n_edge, 1)), dtype=torch.float32),
        "muB": torch.tensor(muB, dtype=torch.float32),
        "quadB": torch.tensor(quadB, dtype=torch.float32),
        "e_ABsr_source": torch.arange(n_edge),
        "e_ABsr_target": torch.arange(n_edge),
        "dR_ang": torch.tensor(dR, dtype=torch.float32),
        "dR_xyz_ang": torch.tensor(dR_xyz, dtype=torch.float32),
    }

    off = model_factory(quadrupole_scale=1.5, elst_include_uQ_QQ=False)
    on = model_factory(quadrupole_scale=1.5, elst_include_uQ_QQ=True)
    assert off.elst_include_uQ_QQ is False
    e_off = off.mtp_elst(**inputs)
    e_on = on.mtp_elst(**inputs)

    # mtp_elst is weight-free, so the two models differ only by the flag.
    assert not torch.allclose(e_off, e_on)
    delta = (e_on - e_off) / 627.509
    expected = reference_uQ_QQ(
        dR_xyz / 0.52917721067,
        muA,
        muB,
        1.5 * quadA,
        1.5 * quadB,
    )
    np.testing.assert_allclose(
        delta.numpy(), expected, rtol=2e-5, atol=1e-9
    )


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_flag_is_reported_in_the_config(model_factory):
    """``get_config`` feeds checkpoint writing, so the flag has to appear there."""
    assert model_factory().get_config()["elst_include_uQ_QQ"] is False
    assert (
        model_factory(elst_include_uQ_QQ=True).get_config()["elst_include_uQ_QQ"]
        is True
    )
