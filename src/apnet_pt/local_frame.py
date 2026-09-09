"""Body-fixed local frames and Racah-normalized real spherical harmonics.

This module supplies the two pieces the MASTIFF exchange anisotropy needs and
that the multipole-guided prefactor did not have.

MASTIFF (Janicki, Van Vleet, Schmidt, *J. Phys. Chem. A* **2023**, 127, 1736)
writes the anisotropic exchange amplitude of Eq. (3) as

.. math::

    A_i(\\theta_i, \\varphi_i)
      = A_{i,\\mathrm{iso}} \\bigl( 1 + \\xi_i \\bigr),
    \\qquad
    \\xi_i = \\sum_{l>0, m} a_{i,lm}
             \\sqrt{\\tfrac{4\\pi}{2l+1}}\\, Y_{lm}(\\theta_i, \\varphi_i),

with :math:`(\\theta_i, \\varphi_i)` the polar angles of the interatomic unit
vector :math:`\\hat r_{ij}` **expressed in a body-fixed frame attached to atom
i**.  Two things about that expression are load-bearing and are the reason this
module exists rather than a two-term reuse of the dipole/quadrupole contraction:

1. the basis is a *free* orthogonal real-spherical-harmonic basis, not a shape
   pinned to the orientation of the atom's own multipoles, and
2. the coefficients are read in a frame that rotates with the molecule, so the
   angular structure is a property of the atom's environment rather than of the
   laboratory axes.

``sqrt(4 pi / (2l+1)) Y_lm`` is the Racah (Stone) normalization, written
:math:`C_{lm}` throughout.  In cartesian form on a unit vector
:math:`(x, y, z)`:

===========  ===========================================
:math:`C`    value
===========  ===========================================
``10``       :math:`z`
``11c``      :math:`x`
``11s``      :math:`y`
``20``       :math:`\\tfrac32 z^2 - \\tfrac12`
``21c``      :math:`\\sqrt3\\, x z`
``21s``      :math:`\\sqrt3\\, y z`
``22c``      :math:`\\tfrac{\\sqrt3}{2}\\, (x^2 - y^2)`
``22s``      :math:`\\sqrt3\\, x y`
===========  ===========================================

Only ``l <= 2`` is provided.  MASTIFF reports that ``l = 3`` bought only a minor
RMSE improvement and excluded it, and every symmetry-allowed coefficient in its
benzene fit is ``l <= 2``.

Parity
------
``e_y`` below is built as ``e_z x e_x`` and is therefore a *pseudo*-vector: under
a global reflection the constructed frame's y axis points the wrong way, so
``phi -> -phi`` and the ``sin(m phi)`` channels (``11s``, ``21s``, ``22s``) flip
sign.  A model carrying those channels distinguishes a dimer from its mirror
image, which exchange repulsion does not.  :data:`PARITY_EVEN_LABELS` is the
subset that is invariant under reflection, and it is the default.  It is also a
superset of every symmetry-allowed coefficient in MASTIFF's benzene fit
(``a_10``, ``a_20`` for the C-inf-v hydrogen; ``a_10``, ``a_20``, ``a_22c`` for
the C-2v carbon), so nothing the paper fits is lost by taking it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = [
    "RACAH_L1L2_LABELS",
    "PARITY_EVEN_LABELS",
    "racah_harmonics_l1l2",
    "CueRacahHarmonics",
    "resolve_harmonics",
    "channel_indices",
    "gram_schmidt_frame",
    "channel_mask",
    "local_angular_xi",
    "LearnedLocalFrame",
    "FRAME_DEGENERACY_EPS",
    "FRAME_COLLINEARITY_EPS",
]

#: Column order of :func:`racah_harmonics_l1l2`.
RACAH_L1L2_LABELS: tuple[str, ...] = (
    "10", "11c", "11s", "20", "21c", "21s", "22c", "22s",
)

#: The reflection-invariant subset -- every ``cos(m phi)`` channel.  See the
#: module docstring's Parity section for why the complement is excluded by
#: default rather than merely discouraged.
PARITY_EVEN_LABELS: tuple[str, ...] = ("10", "11c", "20", "21c", "22c")

#: Below this, a frame axis is treated as undefined rather than normalized.
#: An atom with no intramolecular neighbours has a spherically symmetric
#: environment and gets ``xi = 0``; an atom in a linear environment has no
#: well-defined azimuth and keeps only its ``m = 0`` channels.  Both are
#: handled by zeroing coefficients, never by perturbing the geometry.
FRAME_DEGENERACY_EPS = 1e-6

#: Azimuth degeneracy threshold, applied *relative* to ``|v_x|``.
#:
#: The polar test above can be absolute because ``v_z`` is genuinely zero when
#: an atom has no neighbours.  Collinearity cannot: ``v_x`` is a learned
#: accumulation whose magnitude is arbitrary and drifts during training, so a
#: fixed cutoff means a different angle at every scale.  A residual that is a
#: fraction ``f`` of ``|v_x|`` survived a cancellation of ``1 - f``, leaving a
#: direction whose relative error is about ``float32 eps / f``.  At ``f = 1e-3``
#: that is ~1e-4, which is the accuracy the ``m != 0`` channels are then
#: entitled to; below it the azimuth is rounding noise and gets masked rather
#: than fitted.  Cf. the fixed-atol lesson: a tolerance that does not scale with
#: its operand stops meaning what it was chosen to mean.
FRAME_COLLINEARITY_EPS = 1e-3

_SQRT3 = math.sqrt(3.0)


def channel_indices(labels) -> tuple[int, ...]:
    """Map Racah labels onto columns of :func:`racah_harmonics_l1l2`."""
    out = []
    for label in labels:
        label = str(label).strip().lower()
        if label not in RACAH_L1L2_LABELS:
            raise ValueError(
                f"unknown Racah channel {label!r}; expected one of "
                f"{list(RACAH_L1L2_LABELS)}"
            )
        out.append(RACAH_L1L2_LABELS.index(label))
    if len(set(out)) != len(out):
        raise ValueError("duplicate Racah channel requested")
    if not out:
        raise ValueError("at least one Racah channel is required")
    return tuple(out)


def racah_harmonics_l1l2(unit: torch.Tensor) -> torch.Tensor:
    """Racah-normalized real spherical harmonics for ``l = 1, 2``.

    Parameters
    ----------
    unit
        Unit vectors, shape ``[..., 3]``.  Not renormalized here: the caller
        owns the normalization so the guard against a zero vector stays in one
        place.

    Returns
    -------
    torch.Tensor
        Shape ``[..., 8]`` in :data:`RACAH_L1L2_LABELS` order.
    """
    if unit.shape[-1] != 3:
        raise ValueError(f"expected trailing dimension 3, got {tuple(unit.shape)}")
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    return torch.stack(
        (
            z,
            x,
            y,
            1.5 * z * z - 0.5,
            _SQRT3 * x * z,
            _SQRT3 * y * z,
            0.5 * _SQRT3 * (x * x - y * y),
            _SQRT3 * x * y,
        ),
        dim=-1,
    )


class CueRacahHarmonics(nn.Module):
    """``cuequivariance-torch`` spherical harmonics in the Racah convention.

    ``cuet.SphericalHarmonics`` emits its own normalization and its own ordering
    of the ``l = 1`` and ``l = 2`` irreps.  Rather than hard-code a convention
    that a future release is free to change, the change of basis is *measured*
    at construction against :func:`racah_harmonics_l1l2` on random unit vectors
    in float64 and the residual is asserted at machine precision.  The result is
    an 8x8 constant, so the check costs one fit per process and the forward is
    one matmul on top of the library call.
    """

    #: Fit residual above which the two conventions are not related by a linear
    #: map on the l=1 + l=2 block, which would mean the library's output layout
    #: changed shape rather than basis.
    TOLERANCE = 1e-10

    def __init__(self, *, samples: int = 4096, seed: int = 0):
        super().__init__()
        import cuequivariance_torch as cuet  # noqa: PLC0415  (optional backend)

        self._sh = cuet.SphericalHarmonics([1, 2], normalize=True)
        generator = torch.Generator().manual_seed(seed)
        probe = torch.nn.functional.normalize(
            torch.randn(samples, 3, dtype=torch.float64, generator=generator),
            dim=-1,
        )
        with torch.no_grad():
            raw = self._sh.to(torch.float64)(probe)
        if raw.shape[-1] != len(RACAH_L1L2_LABELS):
            raise RuntimeError(
                "cuequivariance returned "
                f"{raw.shape[-1]} components for l=1,2; expected "
                f"{len(RACAH_L1L2_LABELS)}"
            )
        target = racah_harmonics_l1l2(probe)
        basis = torch.linalg.lstsq(raw, target).solution
        residual = (raw @ basis - target).abs().max().item()
        if not (residual < self.TOLERANCE):
            raise RuntimeError(
                "cuequivariance spherical harmonics could not be mapped onto "
                f"the Racah convention: residual {residual:.3e} exceeds "
                f"{self.TOLERANCE:.1e}"
            )
        self.register_buffer("basis", basis)
        self.fit_residual = residual

    def forward(self, unit: torch.Tensor) -> torch.Tensor:
        raw = self._sh.to(unit.dtype)(unit.reshape(-1, 3))
        out = raw @ self.basis.to(unit.dtype)
        return out.reshape(*unit.shape[:-1], len(RACAH_L1L2_LABELS))


def resolve_harmonics(backend: str = "auto", **kwargs):
    """Return a callable producing Racah ``l <= 2`` harmonics.

    ``backend`` is one of ``"torch"`` (the closed form above), ``"cuequivariance"``
    (fail if unavailable), or ``"auto"`` (cuequivariance when importable, closed
    form otherwise).  ``auto`` never raises, so a CPU test run and a GPU
    production run take the same code path with the same numbers.
    """
    backend = str(backend).strip().lower()
    if backend not in ("auto", "torch", "cuequivariance"):
        raise ValueError(
            "harmonics backend must be 'auto', 'torch', or 'cuequivariance', "
            f"got {backend!r}"
        )
    if backend == "torch":
        return racah_harmonics_l1l2
    try:
        return CueRacahHarmonics(**kwargs)
    except Exception:
        if backend == "cuequivariance":
            raise
        return racah_harmonics_l1l2


def _arbitrary_perpendicular(e_z: torch.Tensor) -> torch.Tensor:
    """A unit vector perpendicular to ``e_z``, chosen numerically safely.

    Used only where the azimuth is undefined, in which case the answer must not
    depend on this choice: see :func:`channel_mask`, which drops every
    ``m != 0`` channel there.  Cross ``e_z`` with whichever cartesian axis it is
    least aligned with, so the cross product is never near-degenerate.
    """
    axis = torch.zeros_like(e_z)
    axis.scatter_(-1, e_z.abs().argmin(dim=-1, keepdim=True), 1.0)
    perpendicular = torch.linalg.cross(axis, e_z, dim=-1)
    return perpendicular / torch.linalg.vector_norm(
        perpendicular, dim=-1, keepdim=True
    ).clamp_min(FRAME_DEGENERACY_EPS)


def gram_schmidt_frame(
    v_z: torch.Tensor,
    v_x: torch.Tensor,
    eps: float = FRAME_DEGENERACY_EPS,
    collinearity_eps: float = FRAME_COLLINEARITY_EPS,
):
    """Orthonormal right-handed frame from two equivariant vectors.

    Returns ``(frame, z_valid, x_valid)`` where ``frame`` is ``[..., 3, 3]`` with
    the basis vectors as **rows**, so ``frame @ r`` gives the components of ``r``
    in the frame.

    The two degeneracies are handled differently, because they destroy different
    amounts of information:

    ``z_valid`` is false where ``v_z`` is too short to define a polar axis -- an
    atom with no intramolecular neighbours, whose environment is spherically
    symmetric.  Nothing angular survives; the frame is filled with the identity
    and every coefficient is masked.

    ``x_valid`` is false where ``v_x`` is collinear with ``v_z``: a linear
    environment, which has a well-defined *axis* but no well-defined azimuth.
    Collinearity is judged on the *relative* residual
    ``|v_x - (v_x . e_z) e_z| / |v_x|``, because ``v_x`` carries no fixed scale.
    Here ``e_z`` is kept, ``e_x`` is filled with an arbitrary perpendicular, and
    only the ``m != 0`` coefficients are masked.  Since ``C_10`` and ``C_20``
    depend on ``r`` solely through ``e_z . r``, the arbitrary choice cannot leak
    into the energy, and the surviving channels stay equivariant.  This is what
    reproduces MASTIFF's C-inf-v hydrogen -- ``a_10`` and ``a_20`` only -- from
    the geometry rather than from a per-type symmetry table.

    Degenerate rows never produce a NaN and the returned frame is always a
    proper rotation, but the mask is not applied here: use :func:`channel_mask`.
    """
    z_norm = torch.linalg.vector_norm(v_z, dim=-1, keepdim=True)
    z_valid = (z_norm > eps).squeeze(-1)
    e_z = v_z / z_norm.clamp_min(eps)
    e_z = torch.where(
        z_valid.unsqueeze(-1),
        e_z,
        torch.tensor([0.0, 0.0, 1.0], dtype=e_z.dtype, device=e_z.device).expand_as(e_z),
    )
    residual = v_x - (v_x * e_z).sum(dim=-1, keepdim=True) * e_z
    x_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    # Relative, not absolute: see `FRAME_COLLINEARITY_EPS`.
    x_scale = torch.linalg.vector_norm(v_x, dim=-1, keepdim=True).clamp_min(eps)
    x_valid = z_valid & (x_norm > collinearity_eps * x_scale).squeeze(-1)
    e_x = torch.where(
        x_valid.unsqueeze(-1),
        residual / x_norm.clamp_min(eps),
        _arbitrary_perpendicular(e_z),
    )
    # One re-orthogonalization pass.  Whatever cancellation the projection
    # above went through, `e_x` comes back perpendicular to `e_z` to machine
    # precision -- and it has to, because a frame that is only orthogonal to
    # 5e-3 is not a rotation, and the rotation invariance of `xi` is the
    # property the entire construction rests on.  "Twice is enough" is exact
    # here: the second pass removes a component of size O(eps), not O(1).
    e_x = e_x - (e_x * e_z).sum(dim=-1, keepdim=True) * e_z
    e_x = e_x / torch.linalg.vector_norm(
        e_x, dim=-1, keepdim=True
    ).clamp_min(eps)
    e_y = torch.linalg.cross(e_z, e_x, dim=-1)
    frame = torch.stack((e_x, e_y, e_z), dim=-2)
    return frame, z_valid, x_valid


def channel_mask(
    z_valid: torch.Tensor,
    x_valid: torch.Tensor,
    active: tuple[int, ...],
) -> torch.Tensor:
    """Which of the ``active`` channels the frame at each atom actually defines.

    Shape ``[..., len(active)]``.  Applying this is not optional: an unmasked
    ``m != 0`` coefficient read in an arbitrary azimuth breaks rotational
    equivariance of the energy, and an unmasked coefficient of any order on an
    atom with no neighbours reads the identity frame, which breaks it too.
    """
    m_zero = torch.tensor(
        [RACAH_L1L2_LABELS[i].endswith("0") for i in active],
        device=z_valid.device,
    )
    allowed = torch.where(x_valid.unsqueeze(-1), True, m_zero)
    return allowed & z_valid.unsqueeze(-1)


def local_angular_xi(
    coefficients: torch.Tensor,
    frame: torch.Tensor,
    z_valid: torch.Tensor,
    x_valid: torch.Tensor,
    unit_external: torch.Tensor,
    active: tuple[int, ...],
    harmonics=racah_harmonics_l1l2,
) -> torch.Tensor:
    """MASTIFF's :math:`\\xi = \\sum_{lm} a_{lm} C_{lm}(\\hat r\\ \\text{in frame})`.

    ``coefficients`` is ``[E, len(active)]`` and ``unit_external`` is ``[E, 3]``
    in the laboratory frame; ``frame``/``z_valid``/``x_valid`` are gathered onto
    the same ``E`` edges.  Masking is applied here so a caller cannot forget it.
    """
    local = torch.einsum("eij,ej->ei", frame, unit_external)
    values = harmonics(local).index_select(
        -1, torch.tensor(active, device=local.device)
    )
    mask = channel_mask(z_valid, x_valid, active)
    return (coefficients * values * mask.to(values.dtype)).sum(dim=-1)


def _cosine_cutoff(distance: torch.Tensor, r_cut: float) -> torch.Tensor:
    """Smooth 1 -> 0 taper so an atom entering ``r_cut`` does not jolt the frame."""
    scaled = (distance / float(r_cut)).clamp(max=1.0)
    return 0.5 * (torch.cos(math.pi * scaled) + 1.0)


class _BesselBasis(nn.Module):
    """Radial basis for the frame weights, self-contained by design.

    ``CliffClassicalNN`` reads frozen atom-model hidden states and owns no
    distance layer, so this module cannot borrow one from its host head.  It is
    the same spherical-Bessel form the rest of the package uses -- learnable
    frequencies times a smooth cutoff -- written here so ``local_frame`` depends
    on nothing but torch and stays unit-testable in isolation.
    """

    def __init__(self, n_rbf: int, r_cut: float):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.r_cut = float(r_cut)
        self.frequencies = nn.Parameter(
            math.pi * torch.arange(1, self.n_rbf + 1, dtype=torch.get_default_dtype())
        )

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        scaled = (distance / self.r_cut).unsqueeze(-1)
        frequencies = self.frequencies.to(distance.dtype)
        return _cosine_cutoff(distance, self.r_cut).unsqueeze(-1) * torch.sin(
            frequencies * scaled
        )


class LearnedLocalFrame(nn.Module):
    """Predict a per-atom body-fixed frame from the atom's own neighbourhood.

    MASTIFF assigns frames by hand: an AMOEBA-style rule (Z-then-X, Bisector,
    Z-Bisect, Threefold, Z-Only) is written into the force-field XML per particle
    type, naming the specific neighbours that define the axes.  That is workable
    for one molecule with typed atoms and is not available here, where the model
    must frame an arbitrary dimer with no typing pass.

    The construction used instead is a learned pair of equivariant vectors

    .. math::

        v^{(a)}_i = \\sum_{j \\in N(i)} f_{\\rm cut}(d_{ij})\\,
                    w^{(a)}(h_i, h_j, d_{ij})\\, \\hat r_{ij},
        \\qquad a \\in \\{z, x\\},

    with the weights ``w`` read off invariant features, followed by
    Gram-Schmidt.  This is equivariant by construction (the only vectors in the
    sum are :math:`\\hat r_{ij}`), invariant to neighbour permutation (it is a
    sum), smooth in the geometry (the cutoff tapers), and defined for any
    neighbour count -- and it can represent the hand-written rules, since
    Z-then-X is one weight on one neighbour.

    It also reproduces MASTIFF's *symmetry reduction* without a symmetry table.
    In a linear environment every :math:`\\hat r_{ij}` is collinear, so
    :math:`v^{(x)}` is parallel to :math:`v^{(z)}`, ``x_valid`` is false, and only
    the ``m = 0`` coefficients survive -- exactly the C-inf-v hydrogen of the
    paper's benzene fit.  With no neighbours at all the environment is spherical
    and every coefficient is dropped.
    """

    def __init__(
        self,
        *,
        n_embed: int,
        n_message: int,
        n_rbf: int,
        n_neuron: int,
        r_cut: float,
    ):
        super().__init__()
        self.n_embed = int(n_embed)
        self.n_message = int(n_message)
        self.n_rbf = int(n_rbf)
        self.r_cut = float(r_cut)
        self.radial = _BesselBasis(self.n_rbf, self.r_cut)
        width = 2 * self.n_embed * (self.n_message + 1) + self.n_rbf
        hidden = max(int(n_neuron), 1)
        self.weight_mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 1), 2),
        )

    def forward(
        self,
        h_list: torch.Tensor,
        unit_ij: torch.Tensor,
        distance: torch.Tensor,
        e_source: torch.Tensor,
        e_target: torch.Tensor,
        n_atoms: int,
    ):
        """Build frames for ``n_atoms`` atoms from an intramolecular edge list.

        ``h_list`` is ``[n_atoms, n_message + 1, n_embed]``; ``unit_ij`` points
        from ``e_source`` to ``e_target`` and the frame is accumulated onto
        ``e_source``.
        """
        flat = h_list.reshape(h_list.size(0), -1)
        features = torch.cat(
            (
                flat.index_select(0, e_source),
                flat.index_select(0, e_target),
                self.radial(distance),
            ),
            dim=-1,
        )
        weights = self.weight_mlp(features)
        weights = weights * _cosine_cutoff(distance, self.r_cut).unsqueeze(-1)
        contribution = weights.unsqueeze(-1) * unit_ij.unsqueeze(-2)
        accumulated = torch.zeros(
            (n_atoms, 2, 3), dtype=contribution.dtype, device=contribution.device
        )
        accumulated = accumulated.index_add(0, e_source, contribution)
        return gram_schmidt_frame(accumulated[:, 0, :], accumulated[:, 1, :])
