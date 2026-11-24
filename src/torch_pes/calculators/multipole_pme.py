"""Smooth Particle Mesh Ewald (SPME) for atomic multipoles up to rank 2.

Implements :math:`\\mathcal{O}(N \\log N)` reciprocal-space electrostatics,
permanent electric field, and induced dipole-dipole matrix-vector products for
arbitrary triclinic periodic boxes using Cardinal B-splines (Essmann et al.
1995; Sagui, Pedersen & Darden 2004).

Key performance features:

* **Single FFT for energy and permanent field**: spreading ``(q, mu, Theta)``
  onto one real 3D mesh and applying one ``rfftn``/``irfftn`` produces both
  the exact smooth multipole reciprocal energy and the exact permanent electric
  field at every atom.
* **Factored 3D spreading & gathering**: groups stencil terms along the
  z-axis, cutting 3D stencil FLOPs by ~3x for multipole spreading and ~2x for
  field gathering.
* **Reusable B-spline state**: :meth:`MultipolePME.prepare` precomputes grid
  indices, B-spline weights/derivatives, fractional Jacobian, and reciprocal
  influence kernel once per geometry; every polarization CG iteration reuses
  them directly in :meth:`MultipolePME.dipole_matvec`.
* **Compile-friendly**: static integer grid shapes ``(K1, K2, K3)``, flat 1D
  ``index_add``, zero host syncs, zero graph breaks under
  ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..lib.bsplines import bspline_modulus_sq_3d, bspline_weights
from .multipole_ewald import (
    dipole_self_field_coeff,
    multipole_background_energy,
    multipole_self_energy,
)


@dataclass
class SPMEState:
    """Precomputed geometry- and cell-dependent SPME mesh state for one step.

    Shared across permanent electrostatics, permanent electric field, and all
    CG polarization iterations at a fixed geometry.
    """

    ns_mesh: tuple[int, int, int]
    order: int
    volume: Tensor
    jacobian: Tensor  # (3, 3): du / dR = cell^{-1} @ diag(K1, K2, K3)
    flat_idx: Tensor  # (N * p^3,) int64 flat 3D mesh indices
    w0: Tensor  # (N, 3, p)
    w1: Tensor  # (N, 3, p)
    w2: Tensor | None  # (N, 3, p) when max_rank >= 2
    k_kernel: Tensor  # (K1, K2, K3 // 2 + 1) real influence kernel C_hat


def _cell_inverse(cell: Tensor) -> Tensor:
    if cell.is_cuda:
        return torch.linalg.inv_ex(cell)[0]
    return torch.linalg.inv(cell)


def _mesh_kvectors_sq(cell: Tensor, ns_mesh: tuple[int, int, int]) -> Tensor:
    """Squared norm ``|k|^2`` on the ``rfftn`` grid ``(K1, K2, K3 // 2 + 1)``."""
    k1, k2, k3 = ns_mesh
    dtype, device = cell.dtype, cell.device
    inv_cell = _cell_inverse(cell)
    recip = 2.0 * math.pi * inv_cell.T  # rows b1, b2, b3

    f1 = torch.fft.fftfreq(k1, dtype=dtype, device=device) * k1
    f2 = torch.fft.fftfreq(k2, dtype=dtype, device=device) * k2
    f3 = torch.fft.rfftfreq(k3, dtype=dtype, device=device) * k3

    kx = f1[:, None] * recip[0][None, :]
    ky = f2[:, None] * recip[1][None, :]
    kz = f3[:, None] * recip[2][None, :]
    kvec = kx[:, None, None, :] + ky[None, :, None, :] + kz[None, None, :, :]
    return (kvec * kvec).sum(dim=-1)


class MultipolePME(torch.nn.Module):
    """Smooth Particle Mesh Ewald (SPME) calculator for rank-0/1/2 multipoles.

    Parameters
    ----------
    alpha : float
        Ewald splitting parameter (inverse length).
    ns_mesh : tuple[int, int, int]
        Static 3D FFT grid dimensions ``(K1, K2, K3)``. Keeping this as a
        Python tuple of ints guarantees static shapes and zero graph breaks
        under ``torch.compile``.
    order : int
        Cardinal B-spline order ``p``. Default ``6`` (``M_6''`` is ``C^2``,
        forces ``M_6'''`` are ``C^1`` for quadrupoles). Use ``4`` or ``5``
        when only charges/dipoles are needed.
    prefactor : float
        Overall unit prefactor (default ``1.0`` for atomic units).
    """

    def __init__(
        self,
        alpha: float,
        ns_mesh: tuple[int, int, int],
        order: int = 6,
        prefactor: float = 1.0,
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if order < 4:
            raise ValueError(f"order must be >= 4 for multipolar SPME, got {order}")
        self.alpha = float(alpha)
        self.ns_mesh = (int(ns_mesh[0]), int(ns_mesh[1]), int(ns_mesh[2]))
        self.order = int(order)
        self.prefactor = float(prefactor)
        # Precompute B-spline Fourier modulus |b(m)|^2 (cell-independent!)
        self.register_buffer(
            "mod_sq",
            bspline_modulus_sq_3d(self.ns_mesh, self.order, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "shifts",
            torch.arange(self.order, dtype=torch.long),
            persistent=False,
        )

    def kspace_kernel(self, cell: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(C_hat, volume)`` on the ``rfftn`` grid."""
        k_sq = _mesh_kvectors_sq(cell, self.ns_mesh)
        volume = torch.abs(torch.linalg.det(cell))
        safe_k_sq = torch.where(k_sq > 0, k_sq, torch.ones_like(k_sq))
        g_k = torch.where(
            k_sq > 0,
            (4.0 * math.pi / safe_k_sq)
            * torch.exp(-safe_k_sq / (4.0 * self.alpha * self.alpha)),
            torch.zeros_like(k_sq),
        )
        mod_sq = self.mod_sq.to(dtype=cell.dtype, device=cell.device)
        c_hat = (mod_sq * g_k) / volume
        return c_hat, volume

    def prepare(
        self,
        positions: Tensor,
        cell: Tensor,
        deriv_order: int = 2,
    ) -> SPMEState:
        """Precompute B-spline weights, flat 3D mesh indices, Jacobian, and
        reciprocal influence kernel for ``(positions, cell)``."""
        k1, k2, k3 = self.ns_mesh
        p = self.order
        dtype, device = positions.dtype, positions.device
        inv_cell = _cell_inverse(cell)
        k_vec = torch.tensor([k1, k2, k3], dtype=dtype, device=device)
        # J = H^{-1} diag(K1, K2, K3) so u = R @ J and d/dR = J @ d/du
        jacobian = inv_cell * k_vec[None, :]
        u = positions @ jacobian  # (N, 3)

        i0 = torch.floor(u)
        delta = u - i0  # (N, 3) in [0, 1)
        i0_l = i0.detach().long()

        ws = bspline_weights(delta, p, deriv_order=deriv_order)
        w0 = ws[0]  # (N, 3, p)
        w1 = ws[1] if deriv_order >= 1 else None
        w2 = ws[2] if deriv_order >= 2 else None

        shifts = self.shifts.to(device=device)  # (p,)
        # Grid indices n = (i0 - k) mod K_a for k = 0 .. p-1, per-axis scalar mod
        # (avoids Inductor CPU vector-remainder bug on length-3 tensors)
        ix = torch.remainder(i0_l[:, 0:1] - shifts[None, :], k1)  # (N, p)
        iy = torch.remainder(i0_l[:, 1:2] - shifts[None, :], k2)  # (N, p)
        iz = torch.remainder(i0_l[:, 2:3] - shifts[None, :], k3)  # (N, p)

        flat_idx = (
            ((ix[:, :, None, None] * k2) + iy[:, None, :, None]) * k3
            + iz[:, None, None, :]
        ).reshape(-1)

        c_hat, volume = self.kspace_kernel(cell)
        assert w1 is not None
        return SPMEState(
            ns_mesh=self.ns_mesh,
            order=p,
            volume=volume,
            jacobian=jacobian,
            flat_idx=flat_idx,
            w0=w0,
            w1=w1,
            w2=w2,
            k_kernel=c_hat,
        )

    def _convolve(self, mesh: Tensor, state: SPMEState) -> Tensor:
        """Convolve real 3D mesh ``(K1, K2, K3)`` with ``state.k_kernel``."""
        mesh_hat = torch.fft.rfftn(mesh, norm="backward")
        return torch.fft.irfftn(
            mesh_hat * state.k_kernel, s=state.ns_mesh, norm="forward"
        )

    def spread_multipoles(
        self,
        state: SPMEState,
        charges: Tensor,
        dipoles: Tensor | None = None,
        quadrupoles: Tensor | None = None,
    ) -> Tensor:
        """Spread ``(q, mu, Theta)`` onto a single 3D mesh ``(K1, K2, K3)``
        using z-factored stencil contraction."""
        k1, k2, k3 = state.ns_mesh
        w0x, w0y, w0z = state.w0[:, 0, :], state.w0[:, 1, :], state.w0[:, 2, :]
        w1x, w1y, w1z = state.w1[:, 0, :], state.w1[:, 1, :], state.w1[:, 2, :]

        # (N, p, p) xy products
        xy00 = w0x[:, :, None] * w0y[:, None, :]
        a0 = charges[:, None, None] * xy00
        a1 = None
        a2 = None

        if dipoles is not None:
            # mu_bar = mu @ J
            mu_bar = dipoles @ state.jacobian  # (N, 3)
            xy10 = w1x[:, :, None] * w0y[:, None, :]
            xy01 = w0x[:, :, None] * w1y[:, None, :]
            a0 = a0 + mu_bar[:, 0:1, None] * xy10 + mu_bar[:, 1:2, None] * xy01
            a1 = mu_bar[:, 2:3, None] * xy00

        if quadrupoles is not None:
            assert state.w2 is not None
            w2x, w2y, w2z = state.w2[:, 0, :], state.w2[:, 1, :], state.w2[:, 2, :]
            # Q_tilde = Theta / 3; Q_bar = J^T @ Q_tilde @ J
            jt_q = torch.einsum("ab,nbc->nac", state.jacobian.T, quadrupoles) / 3.0
            q_bar = torch.einsum("nab,bc->nac", jt_q, state.jacobian)
            if dipoles is None:
                xy10 = w1x[:, :, None] * w0y[:, None, :]
                xy01 = w0x[:, :, None] * w1y[:, None, :]
            xy20 = w2x[:, :, None] * w0y[:, None, :]
            xy02 = w0x[:, :, None] * w2y[:, None, :]
            xy11 = w1x[:, :, None] * w1y[:, None, :]
            a0 = (
                a0
                + q_bar[:, 0, 0][:, None, None] * xy20
                + q_bar[:, 1, 1][:, None, None] * xy02
                + (q_bar[:, 0, 1] + q_bar[:, 1, 0])[:, None, None] * xy11
            )
            a1_q = (
                (q_bar[:, 0, 2] + q_bar[:, 2, 0])[:, None, None] * xy10
                + (q_bar[:, 1, 2] + q_bar[:, 2, 1])[:, None, None] * xy01
            )
            a1 = a1_q if a1 is None else (a1 + a1_q)
            a2 = q_bar[:, 2, 2][:, None, None] * xy00

        weights = a0[:, :, :, None] * w0z[:, None, None, :]
        if a1 is not None:
            weights = weights + a1[:, :, :, None] * w1z[:, None, None, :]
        if a2 is not None:
            assert state.w2 is not None
            weights = weights + a2[:, :, :, None] * state.w2[:, 2, :][:, None, None, :]

        mesh_flat = charges.new_zeros(k1 * k2 * k3)
        mesh_flat = mesh_flat.index_add(0, state.flat_idx, weights.reshape(-1))
        return mesh_flat.reshape(k1, k2, k3)

    def spread_dipoles(self, state: SPMEState, dipoles: Tensor) -> Tensor:
        """Fast rank-1 dipole spreading onto a 3D mesh ``(K1, K2, K3)``."""
        k1, k2, k3 = state.ns_mesh
        w0x, w0y, w0z = state.w0[:, 0, :], state.w0[:, 1, :], state.w0[:, 2, :]
        w1x, w1y, w1z = state.w1[:, 0, :], state.w1[:, 1, :], state.w1[:, 2, :]
        p_bar = dipoles @ state.jacobian  # (N, 3)

        b0 = (
            p_bar[:, 0:1, None] * (w1x[:, :, None] * w0y[:, None, :])
            + p_bar[:, 1:2, None] * (w0x[:, :, None] * w1y[:, None, :])
        )
        b1 = p_bar[:, 2:3, None] * (w0x[:, :, None] * w0y[:, None, :])
        weights = (
            b0[:, :, :, None] * w0z[:, None, None, :]
            + b1[:, :, :, None] * w1z[:, None, None, :]
        )
        mesh_flat = dipoles.new_zeros(k1 * k2 * k3)
        mesh_flat = mesh_flat.index_add(0, state.flat_idx, weights.reshape(-1))
        return mesh_flat.reshape(k1, k2, k3)

    def gather_gradient(self, state: SPMEState, phi_mesh: Tensor) -> Tensor:
        """Gather Cartesian potential gradient :math:`\\nabla_{\\mathbf{R}_i} \\phi`
        from convolved mesh ``phi_mesh`` using factored contraction.

        Note: the electric field is ``-gather_gradient(state, phi_mesh)``.
        """
        n_atoms = state.w0.shape[0]
        p = state.order
        w0x, w0y, w0z = state.w0[:, 0, :], state.w0[:, 1, :], state.w0[:, 2, :]
        w1x, w1y, w1z = state.w1[:, 0, :], state.w1[:, 1, :], state.w1[:, 2, :]

        phi = phi_mesh.reshape(-1)[state.flat_idx].reshape(n_atoms, p, p, p)
        # Contract z first: (N, p, p)
        pz0 = (phi * w0z[:, None, None, :]).sum(dim=-1)
        pz1 = (phi * w1z[:, None, None, :]).sum(dim=-1)
        # Contract y next: (N, p)
        py0_z0 = (pz0 * w0y[:, None, :]).sum(dim=-1)
        py1_z0 = (pz0 * w1y[:, None, :]).sum(dim=-1)
        py0_z1 = (pz1 * w0y[:, None, :]).sum(dim=-1)
        # Contract x last: (N,)
        gx = (py0_z0 * w1x).sum(dim=-1)
        gy = (py1_z0 * w0x).sum(dim=-1)
        gz = (py0_z1 * w0x).sum(dim=-1)
        grad_u = torch.stack([gx, gy, gz], dim=-1)  # (N, 3)
        return grad_u @ state.jacobian.T

    def reciprocal_energy_and_field(
        self,
        positions: Tensor,
        cell: Tensor,
        charges: Tensor,
        dipoles: Tensor | None = None,
        quadrupoles: Tensor | None = None,
        state: SPMEState | None = None,
    ) -> tuple[Tensor, Tensor, SPMEState]:
        """Compute reciprocal-space multipole energy (minus self & background),
        permanent electric field (minus self-field), and reusable :class:`SPMEState`.

        Uses a **single** 3D ``rfftn``/``irfftn`` pair for both energy and field.
        """
        deriv_order = 2 if quadrupoles is not None else 1
        if state is None:
            state = self.prepare(positions, cell, deriv_order=deriv_order)

        q_mesh = self.spread_multipoles(state, charges, dipoles, quadrupoles)
        phi_mesh = self._convolve(q_mesh, state)

        e_rec = 0.5 * (q_mesh * phi_mesh).sum()
        e_rec = (
            e_rec
            - multipole_self_energy(self.alpha, charges, dipoles, quadrupoles)
            + multipole_background_energy(self.alpha, state.volume, charges)
        )

        field = -self.gather_gradient(state, phi_mesh)
        if dipoles is not None:
            field = field + dipole_self_field_coeff(self.alpha) * dipoles

        return self.prefactor * e_rec, self.prefactor * field, state

    def reciprocal_dipole_matvec(
        self,
        dipoles: Tensor,
        state: SPMEState,
    ) -> Tensor:
        """Reciprocal-space induced dipole-dipole matvec
        :math:`(T^{\\mathrm{pol,rec}} \\mathbf{p})_i` (minus self-term) reusing
        precomputed ``state`` (1 spread, 1 FFT/IFFT, 1 gather)."""
        p_mesh = self.spread_dipoles(state, dipoles)
        psi_mesh = self._convolve(p_mesh, state)
        t_p = self.gather_gradient(state, psi_mesh)
        t_p = t_p - dipole_self_field_coeff(self.alpha) * dipoles
        return self.prefactor * t_p

    def reciprocal_dipole_energy(
        self,
        dipoles: Tensor,
        state: SPMEState,
    ) -> Tensor:
        """Differentiable reciprocal-space dipole-dipole energy
        :math:`\\frac{1}{2} \\sum_{k \\ne l} \\mathbf{p}_k^T T^{\\mathrm{pol,rec}}_{kl} \\mathbf{p}_l`
        (minus self-energy), using one spread + ``rfftn``/``irfftn``."""
        p_mesh = self.spread_dipoles(state, dipoles)
        psi_mesh = self._convolve(p_mesh, state)
        e_rec = 0.5 * (p_mesh * psi_mesh).sum()
        e_self = (0.5 * dipole_self_field_coeff(self.alpha)) * (dipoles * dipoles).sum()
        return self.prefactor * (e_rec - e_self)


__all__ = ["MultipolePME", "SPMEState"]
