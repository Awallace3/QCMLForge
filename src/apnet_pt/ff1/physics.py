"""FF-1 pair physics in Hartree atomic units (proposal Eqs. 10-19).

All functions take per-pair tensors already gathered for intermonomer pairs
``(i, j)`` with ``rhat`` pointing from i to j and ``r`` in bohr.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import gordon1

A_POL = 0.385
C_POL = (48.0 * A_POL) ** (1.0 / 3.0)
EPS_POL = 1e-6
C_CT, C_NUC, R_CAP, ALPHA_CAP = 0.80, 0.25, 0.20, 0.35


# --- exchange ----------------------------------------------------------------

def overlap_decay(x: Tensor, y: Tensor) -> Tensor:
    """P_ov(x, y) exp(-x): exact two-center 1s^2 overlap prefactor times decay.

    ``P_ov = (1 + x) sinh(y)/y + x^2 (y cosh y - sinh y)/y^3``, evaluated by
    its even Taylor series in y for |y| < 1 (so it is smooth at y = 0).
    Requires x >= |y| (true when both exponents are positive).
    """
    ya = y.abs()
    small = ya < 1.0
    y2 = torch.where(small, y * y, torch.zeros_like(y))
    s1 = torch.zeros_like(y)
    s2 = torch.zeros_like(y)
    for n in range(9, -1, -1):
        f = 1.0 / math.factorial(2 * n + 1)
        s1 = s1 * y2 + f
        s2 = s2 * y2 + f / (2 * n + 3)
    series = ((1.0 + x) * s1 + x * x * s2) * torch.exp(-x)
    ys = torch.where(small, torch.ones_like(ya), ya)
    ep, em = torch.exp(ys - x), torch.exp(-ys - x)
    sh, ch = 0.5 * (ep - em), 0.5 * (ep + em)
    closed = (1.0 + x) * sh / ys + x * x * (ys * ch - sh) / ys**3
    return torch.where(small, series, closed)


def xi(u: Tensor, Q: Tensor, n: Tensor) -> Tensor:
    """xi(n) = u . n + n^T Q n (per pair)."""
    return (u * n).sum(-1) + torch.einsum("pa,pab,pb->p", n, Q, n)


def transverse_invariants(rhat: Tensor, ui: Tensor, Qi: Tensor, uj: Tensor,
                          Qj: Tensor) -> Tensor:
    """Swap-symmetric SO(2)-complete l <= 2 invariants omega (P, 4)."""
    eye = torch.eye(3, dtype=rhat.dtype, device=rhat.device)
    proj = eye - rhat[:, :, None] * rhat[:, None, :]
    pu_i = torch.einsum("pab,pb->pa", proj, ui)
    pu_j = torch.einsum("pab,pb->pa", proj, uj)
    qr_i = torch.einsum("pab,pb->pa", proj, torch.einsum("pab,pb->pa", Qi, rhat))
    qr_j = torch.einsum("pab,pb->pa", proj, torch.einsum("pab,pb->pa", Qj, -rhat))
    w11 = (pu_i * pu_j).sum(-1)
    w12 = (pu_i * qr_j).sum(-1) + (pu_j * qr_i).sum(-1)
    w221 = (qr_i * qr_j).sum(-1)

    def perp(Q):
        pq = proj @ Q @ proj
        tr = torch.diagonal(pq, dim1=1, dim2=2).sum(-1)
        return pq - 0.5 * tr[:, None, None] * proj

    w222 = (perp(Qi) * perp(Qj)).sum((-1, -2))
    return torch.stack([w11, w12, w221, w222], dim=-1)


def exchange_pairs(r, rhat, A_i, A_j, B_ij, B_ji, pre_i, pre_j, omega, kappa,
                   s_cut=None):
    """E_exch per pair (E_h) >= 0 (Eq. 12)."""
    bbar = 0.5 * (B_ij + B_ji)
    delta = 0.5 * (B_ij - B_ji)
    ang = pre_i + pre_j + (omega * kappa).sum(-1)
    e = A_i * A_j / bbar**3 * torch.exp(ang) * overlap_decay(bbar * r, delta * r)
    return e if s_cut is None else e * s_cut


# --- charge transfer ---------------------------------------------------------

def charge_transfer_pairs(r, e_exch, z_i, z_j, gamma_sum, Bct_ij, Bct_ji,
                          eps_ct: float = 1e-30, s_cut=None):
    """Wall-bounded donor-acceptor CT per pair (E_h) <= 0 (Eq. 16)."""
    bbar = 0.5 * (Bct_ij + Bct_ji)
    delta = 0.5 * (Bct_ij - Bct_ji)
    e_t = gamma_sum / bbar**3 * overlap_decay(bbar * r, delta * r)
    if s_cut is not None:
        e_t = e_t * s_cut
    wall = C_CT * e_exch + C_NUC * z_i * z_j / torch.sqrt(r * r + R_CAP**2) \
        * torch.exp(-ALPHA_CAP * r)
    m = e_t + wall + eps_ct
    den = ((e_t / m) ** 4 + (wall / m) ** 4 + (eps_ct / m) ** 4) ** 0.25
    return -e_t * (wall / m) / den


# --- dispersion --------------------------------------------------------------

def dispersion_pairs(r, a_i, a_j, eta_i, eta_j, r42_i, r42_j, af_i, af_j,
                     a1, a2, s8, eps_q=1e-6, eps_c6=1e-12, s_cut=None):
    """(E_disp per pair (E_h) <= 0, C6_ij) with BJ damping (Eqs. 17-19)."""
    c6 = a_i * a_j * 2.0 * eta_i * eta_j / (eta_i + eta_j)
    ratio = ((a_i + eps_q) * (a_j + eps_q) / ((af_i + eps_q) * (af_j + eps_q)))
    c8 = 3.0 * c6 * r42_i * r42_j * ratio ** (1.0 / 3.0)
    r0 = a1 * torch.sqrt((c8 + eps_c6) / (c6 + eps_c6)) + a2
    e = -(c6 / (r**6 + r0**6) + s8 * c8 / (r**8 + r0**8))
    return (e if s_cut is None else e * s_cut), c6


# --- polarization ------------------------------------------------------------

def beta_pol(alpha_iso: Tensor) -> Tensor:
    return C_POL * (alpha_iso + EPS_POL) ** (-1.0 / 3.0)


def dense_layout(atom_dimer: Tensor, n_dimer: int):
    """(local index per atom, max atoms per dimer, pair index lists) for the
    all-pairs k != l SCF of each dimer; atoms must be grouped by dimer."""
    counts = torch.bincount(atom_dimer, minlength=n_dimer)
    offsets = torch.cumsum(counts, 0) - counts
    local = torch.arange(len(atom_dimer), device=atom_dimer.device) - offsets[atom_dimer]
    m = int(counts.max()) if n_dimer else 0
    table = torch.full((n_dimer, m), -1, dtype=torch.long, device=atom_dimer.device)
    table[atom_dimer, local] = torch.arange(len(atom_dimer), device=atom_dimer.device)
    valid = table >= 0
    grid = valid[:, :, None] & valid[:, None, :]
    grid &= ~torch.eye(m, dtype=torch.bool, device=atom_dimer.device)[None]
    d, lk, ll = torch.nonzero(grid, as_tuple=True)
    return local, m, (d, lk, ll, table[d, lk], table[d, ll])


def polarization(R_bohr: Tensor, atom_dimer: Tensor, n_dimer: int,
                 sqrt_alpha: Tensor, beta_p: Tensor, field: Tensor,
                 return_matrix: bool = False):
    """Variational Gordon1-whitened polarization per dimer (E_h, <= 0).

    Solves ``(I + K) w = b`` with ``K_kl = a_k^1/2 T_kl a_l^1/2`` over all pairs
    ``k != l`` of each dimer and returns ``U(R, stop_grad(w*))`` (envelope
    theorem: exact forces and parameter gradients).
    """
    local, m, (d, lk, ll, gk, gl) = dense_layout(atom_dimer, n_dimer)
    rvec = R_bohr[gk] - R_bohr[gl]
    r, rhat = gordon1.safe_unit(rvec)
    T = gordon1.dipole_tensor(r, rhat, beta_p[gk], beta_p[gl])
    K = sqrt_alpha[gk] @ T @ sqrt_alpha[gl]
    blocks = R_bohr.new_zeros((n_dimer, m, m, 3, 3))
    blocks = blocks.index_put((d, lk, ll), K)
    A = blocks.permute(0, 1, 3, 2, 4).reshape(n_dimer, 3 * m, 3 * m)
    A = A + torch.eye(3 * m, dtype=A.dtype, device=A.device)
    b_atom = torch.einsum("nab,nb->na", sqrt_alpha, field)
    b = R_bohr.new_zeros((n_dimer, m, 3)).index_put((atom_dimer, local), b_atom)
    b = b.reshape(n_dimer, 3 * m)
    with torch.no_grad():
        w = torch.linalg.solve(A.detach(), b.detach())
    energy = 0.5 * torch.einsum("da,dab,db->d", w, A, w) - (w * b).sum(-1)
    induced = torch.einsum("nab,nb->na", sqrt_alpha,
                           w.reshape(n_dimer, m, 3)[atom_dimer, local])
    out = {"induced_dipole": induced}
    if return_matrix:
        out["matrix"] = A
    return energy, out
