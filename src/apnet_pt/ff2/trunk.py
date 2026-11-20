"""Optional SO3LR-Tauri pieces on FF-1's l <= 2 trunk (FF-3 backbone arms).

Each piece starts as an exact identity on FF-1's trunk (zero-initialised last
layer), so an FF-1/FF-2/MBIS state dict loads with only the new keys missing:

* ``charge_embedding`` (arm E): SpookyNet/SO3LR total-charge embedding added
  to the element embedding, ``h_i^(0) = E_Z + MLP_res(a_i v_s)`` with
  ``a_i = Q_M softplus(<q_Z_i, k_s> / sqrt F) / sum_{j in M} softplus(...)``
  and ``s = [Q_M < 0]``; ``MLP_res``'s last layer is zero.
* ``attention`` (arm A): So3krates' bounded bilinear weight gating FF-1's
  per-edge message ``phi``:
  ``a_ik = f_c(r) (1 + (1/nbar) sum_c RMS(W_q h_i)_c w_ik,c RMS(W_k h_k)_c)``
  with the filter ``w = W_2 silu(W_1 [rbf, |u_k - u_i|^2, |Q_k - Q_i|^2,
  kappa(e_k - e_i)])`` and ``W_2 = 0``.  Heads: ``n_heads`` over the scalar
  message, one over the vector, one over the tensor message.
* ``euclidean_variables`` (arm A-ev): So3krates Euclidean variables, degrees
  l = 1..4, one copy: ``e_i^(l) += (1/nbar) sum_k a_ik^(l) Y_l(r_ik)`` with
  one gate head per degree, and ``h_i += W_e kappa(e_i)``,
  ``kappa_l = |e^(l)|^2 / sqrt(2l + 1)``, ``W_e = 0``.

``nbar`` is the mean intramonomer neighbour count (SO3LR's
``avg_num_neighbors``); 13.0 is the v6 validation store's 4,959,428 directed
edges over 381,791 atoms.  The trunk contract ``(h, u, Q, geo)`` is FF-1's.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.functional import silu, softplus

from ..ff1.trunk import FF1Layer, FF1MonomerTrunk, f_cut, scatter_sum

DEGREES = (1, 2, 3, 4)
EV_SLICES = ((0, 3), (3, 8), (8, 15), (15, 24))  # components of each degree
N_EV = 24
NBAR = 13.0
RMS_EPS = 1e-6


def real_sph(r: Tensor) -> Tensor:
    """Orthonormal real spherical harmonics l = 1..4 (24 components, m = -l..l)
    of the rows of ``r`` (E, 3), written as homogeneous polynomials so a
    zero-length vector maps to 0 for every degree."""
    x, y, z = r.unbind(-1)
    pi = math.pi
    x2, y2, z2 = x * x, y * y, z * z
    r2 = x2 + y2 + z2
    c1 = math.sqrt(3 / (4 * pi))
    out = [
        c1 * y, c1 * z, c1 * x,
        0.5 * math.sqrt(15 / pi) * x * y, 0.5 * math.sqrt(15 / pi) * y * z,
        0.25 * math.sqrt(5 / pi) * (3 * z2 - r2), 0.5 * math.sqrt(15 / pi) * x * z,
        0.25 * math.sqrt(15 / pi) * (x2 - y2),
        0.25 * math.sqrt(35 / (2 * pi)) * y * (3 * x2 - y2),
        0.5 * math.sqrt(105 / pi) * x * y * z,
        0.25 * math.sqrt(21 / (2 * pi)) * y * (5 * z2 - r2),
        0.25 * math.sqrt(7 / pi) * z * (5 * z2 - 3 * r2),
        0.25 * math.sqrt(21 / (2 * pi)) * x * (5 * z2 - r2),
        0.25 * math.sqrt(105 / pi) * z * (x2 - y2),
        0.25 * math.sqrt(35 / (2 * pi)) * x * (x2 - 3 * y2),
        0.75 * math.sqrt(35 / pi) * x * y * (x2 - y2),
        0.75 * math.sqrt(35 / (2 * pi)) * y * z * (3 * x2 - y2),
        0.75 * math.sqrt(5 / pi) * x * y * (7 * z2 - r2),
        0.75 * math.sqrt(5 / (2 * pi)) * y * z * (7 * z2 - 3 * r2),
        (3 / 16) * math.sqrt(1 / pi) * (35 * z2 * z2 - 30 * z2 * r2 + 3 * r2 * r2),
        0.75 * math.sqrt(5 / (2 * pi)) * x * z * (7 * z2 - 3 * r2),
        (3 / 8) * math.sqrt(5 / pi) * (x2 - y2) * (7 * z2 - r2),
        0.75 * math.sqrt(35 / (2 * pi)) * x * z * (x2 - 3 * y2),
        (3 / 16) * math.sqrt(35 / pi) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)),
    ]
    return torch.stack(out, -1)


def degree_contract(ev: Tensor) -> Tensor:
    """(..., 24) -> (..., 4): ``|e^(l)|^2 / sqrt(2l + 1)`` per degree."""
    return torch.stack([(ev[..., a:b] * ev[..., a:b]).sum(-1) / math.sqrt(b - a)
                        for a, b in EV_SLICES], -1)


def degree_expand(a: Tensor) -> Tensor:
    """(E, 4) per-degree gates -> (E, 24), each repeated over its 2l + 1 m."""
    return torch.cat([a[:, k:k + 1].expand(-1, b - lo)
                      for k, (lo, b) in enumerate(EV_SLICES)], 1)


def rms(x: Tensor) -> Tensor:
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + RMS_EPS)


def identity_init_layer(layer: FF1Layer) -> None:
    """Make an FF1Layer an exact identity: zero its update paths (``phi``,
    ``psi_h``'s last layer, ``w_uq``, ``w_qlin``, ``w_q1``; ``w_q2`` starts
    at 0) and cap its increments rather than its state
    (``u + G_rho(du)``, ``G_rho(0) = 0``), so it never re-caps ``u``/``Q``.
    Trained, the layer stays bounded: ``|u|``, ``|Q|`` per channel grow by
    at most ``rho`` (``<= 2 rho`` after one identity layer on a capped
    trunk)."""
    layer.cap_increment = True
    with torch.no_grad():
        for lin in (layer.phi, layer.psi_h[-1]):
            lin.weight.zero_()
            lin.bias.zero_()
        for lin in (layer.w_uq, layer.w_qlin, layer.w_q1):
            lin.weight.zero_()
        layer.w_q2.zero_()


class ChargeEmbedding(nn.Module):
    """SpookyNet/SO3LR total-charge embedding per monomer (output 0 at init)."""

    def __init__(self, n_feat: int, max_z: int = 55):
        super().__init__()
        self.query = nn.Embedding(max_z, n_feat)
        self.key = nn.Embedding(2, n_feat)
        self.value = nn.Embedding(2, n_feat)
        self.res1 = nn.Linear(n_feat, n_feat, bias=False)
        self.res2 = nn.Linear(n_feat, n_feat, bias=False)
        nn.init.zeros_(self.res2.weight)

    def forward(self, z: Tensor, monomer: Tensor, monomer_charge: Tensor) -> Tensor:
        F = self.query.embedding_dim
        sign = (monomer_charge < 0).long()[monomer]
        y = softplus((self.query(z) * self.key(sign)).sum(-1) / math.sqrt(F))
        den = y.new_zeros(monomer_charge.shape).index_add(0, monomer, y)
        a = monomer_charge.to(y.dtype)[monomer] * y / den[monomer]
        x = a[:, None] * self.value(sign)
        return self.res2(silu(self.res1(silu(x))))


class FF3Layer(FF1Layer):
    """FF1Layer with an optional So3krates attention gate and Euclidean
    variables; exactly FF1Layer at init (``filt2 = 0``, ``w_ev = 0``)."""

    def __init__(self, n_feat: int, n_chan: int, n_rbf: int, rho: float,
                 attention: bool = False, euclidean_variables: bool = False,
                 n_heads: int = 4, d_head: int = 32, nbar: float = NBAR):
        super().__init__(n_feat, n_chan, n_rbf, rho)
        if n_feat % n_heads:
            raise ValueError(f"n_feat {n_feat} is not divisible by n_heads {n_heads}")
        self.attention, self.ev = bool(attention), bool(euclidean_variables)
        self.n_heads, self.d_head, self.nbar = n_heads, d_head, float(nbar)
        n_ev = len(DEGREES) if self.ev else 0
        self.n_gate = (n_heads + 2 if self.attention else 0) + n_ev
        if self.n_gate:
            H, D = self.n_gate, d_head
            self.wq = nn.Linear(n_feat, H * D, bias=False)
            self.wk = nn.Linear(n_feat, H * D, bias=False)
            self.filt1 = nn.Linear(n_rbf + 2 * n_chan + n_ev, n_feat)
            self.filt2 = nn.Linear(n_feat, H * D)
            nn.init.zeros_(self.filt2.weight)
            nn.init.zeros_(self.filt2.bias)
        if self.ev:
            self.w_ev = nn.Linear(len(DEGREES), n_feat, bias=False)
            nn.init.zeros_(self.w_ev.weight)

    def gates(self, h, u, Q, ev, src, dst, rbf, fc):
        """(E, n_gate) gates ``f_c (1 + g)``; ``g = 0`` at init."""
        H, D = self.n_gate, self.d_head
        q = rms(self.wq(h).view(-1, H, D))[src]
        k = rms(self.wk(h).view(-1, H, D))[dst]
        du, dq = u[dst] - u[src], Q[dst] - Q[src]
        feats = [rbf, (du * du).sum(1), (dq * dq).sum((1, 2))]
        if self.ev:
            feats.append(degree_contract(ev[dst] - ev[src]))
        w = self.filt2(silu(self.filt1(torch.cat(feats, 1)))).view(-1, H, D)
        g = (q * w * k).sum(-1) / self.nbar
        return fc[:, None] * (1.0 + g)

    def edge_coefficients(self, h, u, Q, ev, src, dst, rbf, fc, m):
        """(ph, pu, pq, gates): FF-1's ``phi`` split, gated per head."""
        F, C = h.shape[1], self.C
        phi = self.phi(silu(m))
        a = self.gates(h, u, Q, ev, src, dst, rbf, fc) if self.n_gate else None
        if self.attention:
            nh = self.n_heads
            ph = (phi[:, :F].view(-1, nh, F // nh) * a[:, :nh, None]).reshape(-1, F)
            pu = phi[:, F:F + 3 * C] * a[:, nh:nh + 1]
            pq = phi[:, F + 3 * C:] * a[:, nh + 1:nh + 2]
        else:
            phi = phi * fc[:, None]
            ph, pu, pq = phi[:, :F], phi[:, F:F + 3 * C], phi[:, F + 3 * C:]
        return ph, pu, pq, a

    def forward(self, h, u, Q, src, dst, rt, rbf, fc, ev=None, ylm=None):
        n = h.shape[0]
        ui, uk, Qi, Qk = u[src], u[dst], Q[src], Q[dst]
        m = self.message(torch.cat([h[src], h[dst], rbf,
                                    self.edge_invariants(ui, uk, Qi, Qk, rt)], 1))
        ph, pu, pq, a = self.edge_coefficients(h, u, Q, ev, src, dst, rbf, fc, m)
        dh = scatter_sum(ph, src, n) + self.psi_h(
            torch.cat([h, self.self_invariants(u, Q)], dim=1))
        if self.ev:
            a_l = degree_expand(a[:, -len(DEGREES):])
            ev = ev + scatter_sum(a_l * ylm, src, n) / self.nbar
            dh = dh + self.w_ev(degree_contract(ev))
        u, Q = self.update_uq(u, Q, uk, Qk, rt, pu, pq, src, n)
        return h + dh, u, Q, ev


class FF3MonomerTrunk(FF1MonomerTrunk):
    """FF1MonomerTrunk with the optional SO3LR pieces (module docstring);
    returns FF-1's ``(h, u, Q, geo)``.  ``monomer``/``monomer_charge`` are
    read only with ``charge_embedding``."""

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_layers: int = 3,
                 n_rbf: int = 16, r_cut: float = 5.0, rho_tr: float = 4.0,
                 max_z: int = 55, charge_embedding: bool = False,
                 attention: bool = False, euclidean_variables: bool = False,
                 nbar: float = NBAR):
        super().__init__(n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr, max_z)
        self.use_ev = bool(euclidean_variables)
        self.layers = nn.ModuleList(
            FF3Layer(n_feat, n_chan, n_rbf, rho_tr, attention, euclidean_variables,
                     nbar=nbar) for _ in range(n_layers))
        self.charge_embed = (ChargeEmbedding(n_feat, max_z) if charge_embedding
                             else None)

    def forward(self, z: Tensor, R: Tensor, edges: Tensor,
                monomer: Tensor | None = None,
                monomer_charge: Tensor | None = None):
        src, dst = edges
        r, rt_len, rt = self.edge_geometry(R, edges)
        fc = f_cut(r, self.r_cut)
        rbf = self.rbf(rt_len)
        n = z.shape[0]
        h = self.embed(z)
        if self.charge_embed is not None:
            h = h + self.charge_embed(z, monomer, monomer_charge.to(h.dtype))
        u = h.new_zeros((n, 3, self.n_chan))
        Q = h.new_zeros((n, 3, 3, self.n_chan))
        ev = h.new_zeros((n, N_EV)) if self.use_ev else None
        ylm = real_sph(rt) if self.use_ev else None
        for layer in self.layers:
            h, u, Q, ev = layer(h, u, Q, src, dst, rt, rbf, fc, ev, ylm)
        return h, u, Q, {"r": r, "rt": rt, "fc": fc, "rbf": rbf}
