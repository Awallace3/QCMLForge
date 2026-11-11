"""Strict l <= 2 O(3)-equivariant monomer trunk (proposal Eqs. 1-3).

Per-atom state ``(h, u, Q)``: ``h`` (N, F) scalars, ``u`` (N, 3, C) polar
vectors and ``Q`` (N, 3, 3, C) traceless symmetric tensors.  Messages run over
intramonomer edges ``(i, k)`` with the smoothed distance
``r~ = sqrt(r^2 + eps_R^2)``, the quartic C^3 cutoff ``(1 - r^2/r_c^2)^4`` and
SiLU MLPs of invariants only, so every update is exactly O(3)-equivariant
(parity -1 for ``u``, +1 for ``Q``).  Each layer caps every vector/tensor
channel with ``G_rho(X) = rho tanh(|X|_eps / rho) X / |X|_eps``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ANG2BOHR = 1.0 / 0.52917721092
EPS_R = 1e-4 / ANG2BOHR  # 1e-4 bohr, in Angstrom
EPS_U = 1e-8


def f_cut(r: Tensor, r_cut: float) -> Tensor:
    """Quartic C^3 cutoff (1 - r^2/r_c^2)^4 for r < r_c, else 0."""
    s = (1.0 - (r / r_cut) ** 2).clamp_min(0.0)
    return s**4


def eps_norm(x: Tensor, dims, eps: float = EPS_U) -> Tensor:
    return torch.sqrt((x * x).sum(dims) + eps**2)


def norm_cap(x: Tensor, rho: float, dims) -> Tensor:
    """O(3)-equivariant analytic norm bound G_rho over ``dims``."""
    n = eps_norm(x, dims, EPS_U)
    scale = rho * torch.tanh(n / rho) / n
    for d in sorted(d % x.dim() for d in dims):
        scale = scale.unsqueeze(d)
    return x * scale


def sym0(a: Tensor) -> Tensor:
    """Symmetric traceless part over dims (1, 2) of (N, 3, 3, ...)."""
    s = 0.5 * (a + a.transpose(1, 2))
    tr = torch.diagonal(s, dim1=1, dim2=2).sum(-1)
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    eye = eye.reshape(1, 3, 3, *([1] * (a.dim() - 3)))
    return s - tr.unsqueeze(1).unsqueeze(1) / 3.0 * eye


def mlp(n_in: int, n_hidden: int, n_out: int, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(width, n_hidden), nn.SiLU()]
        width = n_hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def scatter_sum(src: Tensor, index: Tensor, n: int) -> Tensor:
    out = src.new_zeros((n, *src.shape[1:]))
    return out.index_add(0, index, src)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, r_cut: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, r_cut, n_rbf))
        self.width = r_cut / n_rbf

    def forward(self, r: Tensor) -> Tensor:
        return torch.exp(-(((r[:, None] - self.centers) / self.width) ** 2))


class FF1Layer(nn.Module):
    """One message-passing update of (h, u, Q) (Eq. 3)."""

    def __init__(self, n_feat: int, n_chan: int, n_rbf: int, rho: float):
        super().__init__()
        F, C = n_feat, n_chan
        self.C, self.rho = C, rho
        self.message = mlp(2 * F + n_rbf + 10 * C, F, F, 2)
        self.phi = nn.Linear(F, F + 6 * C)
        self.psi_h = mlp(F + 4 * C, F, F, 2)
        self.w_uq = nn.Linear(C, C, bias=False)
        self.w_qlin = nn.Linear(C, C, bias=False)
        self.w_q2 = nn.Parameter(torch.zeros(C))
        self.w_q1 = nn.Linear(C, C, bias=False)
        self.w_q2v = nn.Linear(C, C, bias=False)

    @staticmethod
    def self_invariants(u: Tensor, Q: Tensor) -> Tensor:
        nu = eps_norm(u, (1,))
        nq = eps_norm(Q, (1, 2))
        uqu = torch.einsum("nac,nabc,nbc->nc", u, Q, u)
        q2 = torch.einsum("nabc,nbdc->nadc", Q, Q)
        q3 = torch.einsum("nabc,nbac->nc", q2, Q)
        return torch.cat([nu, nq, uqu, q3], dim=1)

    def forward(self, h, u, Q, src, dst, rt, rbf, fc):
        n = h.shape[0]
        ui, uk, Qi, Qk = u[src], u[dst], Q[src], Q[dst]
        inv = torch.cat([
            eps_norm(ui, (1,)), eps_norm(uk, (1,)),
            eps_norm(Qi, (1, 2)), eps_norm(Qk, (1, 2)),
            torch.einsum("eac,ea->ec", ui, rt), torch.einsum("eac,ea->ec", uk, rt),
            (ui * uk).sum(1),
            torch.einsum("ea,eabc,eb->ec", rt, Qi, rt),
            torch.einsum("ea,eabc,eb->ec", rt, Qk, rt),
            (Qi * Qk).sum((1, 2)),
        ], dim=1)
        m = self.message(torch.cat([h[src], h[dst], rbf, inv], dim=1))
        phi = self.phi(torch.nn.functional.silu(m)) * fc[:, None]
        C = self.C
        ph, pu, pq = phi[:, :-6 * C], phi[:, -6 * C:-3 * C], phi[:, -3 * C:]
        qr = torch.einsum("eabc,eb->eac", Qk, rt)
        du_e = (pu[:, None, :C] * rt[:, :, None] + pu[:, None, C:2 * C] * uk
                + pu[:, None, 2 * C:] * qr)
        rr = sym0(rt[:, :, None] * rt[:, None, :])
        ur = sym0(uk[:, :, None, :] * rt[:, None, :, None])
        dq_e = (pq[:, None, None, :C] * rr[..., None] + pq[:, None, None, C:2 * C] * ur
                + pq[:, None, None, 2 * C:] * Qk)
        dh = scatter_sum(ph, src, n) + self.psi_h(
            torch.cat([h, self.self_invariants(u, Q)], dim=1))
        du = scatter_sum(du_e, src, n) + self.w_uq(torch.einsum("nabc,nbc->nac", Q, u))
        a, b = self.w_q1(u), self.w_q2v(u)
        dq = (scatter_sum(dq_e, src, n) + self.w_qlin(Q)
              + self.w_q2 * sym0(torch.einsum("nabc,nbdc->nadc", Q, Q))
              + sym0(a[:, :, None, :] * b[:, None, :, :]))
        u = norm_cap(u + du, self.rho, (1,))
        Q = norm_cap(Q + dq, self.rho, (1, 2))
        return h + dh, u, Q


class FF1MonomerTrunk(nn.Module):
    """Equivariant l <= 2 monomer trunk over intramonomer edges.

    Parameters
    ----------
    n_feat, n_chan : int
        Scalar width F and number of vector/tensor channels C.
    n_layers, n_rbf : int
        Message-passing layers T and radial basis size.
    r_cut : float
        Intramonomer cutoff (Angstrom), 5.0 by default.
    rho_tr : float
        Per-layer channel norm cap.
    """

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_layers: int = 3,
                 n_rbf: int = 16, r_cut: float = 5.0, rho_tr: float = 4.0,
                 max_z: int = 55):
        super().__init__()
        self.n_feat, self.n_chan, self.r_cut = n_feat, n_chan, r_cut
        self.embed = nn.Embedding(max_z, n_feat)
        self.rbf = GaussianRBF(n_rbf, r_cut)
        self.layers = nn.ModuleList(
            FF1Layer(n_feat, n_chan, n_rbf, rho_tr) for _ in range(n_layers))

    @staticmethod
    def edge_geometry(R: Tensor, edges: Tensor):
        """(r, r~, unit vector r~_ik) for edges (i, k) with R in Angstrom."""
        src, dst = edges
        d = R[dst] - R[src]
        r2 = (d * d).sum(-1)
        rt = torch.sqrt(r2 + EPS_R**2)
        return torch.sqrt(r2 + 1e-30), rt, d / rt[:, None]

    def forward(self, z: Tensor, R: Tensor, edges: Tensor):
        """(h, u, Q, edge geometry) for atoms ``z`` at ``R`` (Angstrom)."""
        src, dst = edges
        r, rt_len, rt = self.edge_geometry(R, edges)
        fc = f_cut(r, self.r_cut)
        rbf = self.rbf(rt_len)
        n = z.shape[0]
        h = self.embed(z)
        u = h.new_zeros((n, 3, self.n_chan))
        Q = h.new_zeros((n, 3, 3, self.n_chan))
        for layer in self.layers:
            h, u, Q = layer(h, u, Q, src, dst, rt, rbf, fc)
        return h, u, Q, {"r": r, "rt": rt, "fc": fc, "rbf": rbf}
