"""FF-1: physically grounded l <= 2 intermolecular force field (Eqs. 1-22).

``FF1Model(batch)`` returns ``[elst, exch, ind, disp]`` (kcal/mol) per dimer
with ``ind = E_pol + E_CT``.  The batch is a dict:

=================  =========================================================
``z``              (N,) atomic numbers (1..54)
``R``              (N, 3) coordinates, Angstrom; atoms grouped by dimer
``edges``          (2, E) directed intramonomer edges within ``r_cut``
``monomer``        (N,) global monomer index of each atom
``monomer_charge`` (n_mon,) formal monomer charges
``atom_dimer``     (N,) dimer index of each atom
``ai``, ``aj``     (P,) intermonomer pairs (i in A, j in B)
``pair_dimer``     (P,) dimer index of each pair
``n``              number of dimers
``fragment``       optional (N,) fragment index; with ``fragment_charge``
=================  =========================================================

The trunk runs in ``trunk_dtype`` (float32 in production); charges are solved
in float64 and every physics kernel runs in ``physics_dtype``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from . import charges as ch
from . import elements, gordon1, physics
from .trunk import ANG2BOHR, FF1MonomerTrunk, f_cut, mlp, norm_cap

H2KCAL = 627.5095
THETA_MAX = 0.95
RHO_NU = 0.35
SIGMA_MIN = 0.05
EPS_EL = 1e-2
TAU_AUF = 1e-2
EPS_N = 1e-6
EPS_ALPHA = 1e-12
# bond-share logit g_ik in (-8, 4), starting at -6 (p ~ 0.2%: near-neutral prior)
G_CENTER, G_HALF = -2.0, 6.0
G_SHIFT = G_HALF * math.atanh((G_CENTER + 6.0) / G_HALF)
# head name -> (bound on the scalar delta)
SCALAR_HEADS = {"beta": 0.5, "A": 1.5, "B": 0.4, "alpha": 0.7, "eta": 0.5,
                "D": 1.5, "C": 1.5, "eta_B": 4.0}
VECTOR_HEADS = {"pre": 0.35, "B": 0.30, "ex": 0.35, "D": 0.35, "C": 0.35}
TENSOR_HEADS = {"pre": 0.35, "B": 0.30, "ex": 0.35, "D": 0.35, "C": 0.35,
                "alpha": 0.35}
GUESS_NAMES = ("beta", "A", "B", "D", "C")


def default_element_init() -> dict:
    """Element starting tables (bohr^-1, E_h^1/2 bohr^3/2): exchange B0 = the
    valence density exponent, A0 from a generic MASTIFF-like amplitude, and
    CT donor/acceptor amplitudes at ~0.22 A0 (CT ~ 10% of exchange)."""
    beta = elements.table("beta_val")
    a0 = [30.0 * b**1.5 / math.sqrt(H2KCAL) for b in beta]
    return {"beta": beta, "B": list(beta), "A": a0,
            "D": [0.22 * a for a in a0], "C": [0.22 * a for a in a0]}


def phi_tau(u: Tensor, tau: float = TAU_AUF) -> Tensor:
    return 0.5 * (u + torch.sqrt(u * u + tau * tau))


def psi(x: Tensor, gamma: Tensor, eps: float = EPS_N) -> Tensor:
    """(x^2 + eps^2)^(gamma/2) - eps^gamma: autograd-safe x^gamma, psi(0) = 0."""
    return (x * x + eps * eps) ** (0.5 * gamma) - eps**gamma


class FF1Model(nn.Module):
    """FF-1 force field.

    Parameters
    ----------
    n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr
        Trunk hyperparameters (``FF1MonomerTrunk``).
    element_init : dict, optional
        Per-element starting values overriding ``default_element_init()``
        (lists indexed by Z for ``beta``, ``A``, ``B``, ``D``, ``C``).
    trunk_dtype, physics_dtype : torch.dtype
        Precision of the neural trunk and of the physics kernels.
    s_cut : (float, float), optional
        Septic switch window (Angstrom) for condensed phase; None = 1.
    """

    components = ("elst", "exch", "ind", "disp")

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_layers: int = 3,
                 n_rbf: int = 16, r_cut: float = 5.0, rho_tr: float = 4.0,
                 element_init: dict | None = None,
                 trunk_dtype: torch.dtype = torch.float32,
                 physics_dtype: torch.dtype = torch.float64,
                 s_cut: tuple[float, float] | None = None):
        super().__init__()
        self.trunk_dtype, self.physics_dtype = trunk_dtype, physics_dtype
        self.s_cut_window = s_cut
        self.trunk = FF1MonomerTrunk(n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr)
        F, C = n_feat, n_chan
        for name in elements.TABLE_NAMES:
            self.register_buffer(name, torch.tensor(elements.table(name),
                                                    dtype=torch.float64))
        init = {**default_element_init(), **(element_init or {})}
        for name in GUESS_NAMES:
            self.register_buffer(f"log_{name}0", torch.tensor(init[name],
                                                              dtype=torch.float64).log())
        # per-element learned offsets (start at 0) and Aufbau exponents
        self.guess = nn.Embedding(elements.MAX_Z + 1, len(GUESS_NAMES))
        nn.init.zeros_(self.guess.weight)
        self.gamma_raw = nn.Parameter(torch.zeros(elements.MAX_Z + 1, 2))
        self.edge_head = mlp(2 * F + n_rbf, F, 1, 2)
        self.charge_head = mlp(F, F, 3, 2)
        self.prop_head = mlp(F + 4, F, len(SCALAR_HEADS), 2)
        n_vec, n_ten = 1 + len(VECTOR_HEADS), 1 + len(TENSOR_HEADS)
        self.vec_readout = nn.Linear(C, n_vec, bias=False)
        self.ten_readout = nn.Linear(C, n_ten, bias=False)
        for layer in (self.charge_head[-1], self.prop_head[-1], self.edge_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.vec_readout.weight.mul_(0.1)
            self.ten_readout.weight.mul_(0.1)
        self.kappa_ex_raw = nn.Parameter(torch.zeros(4))
        self.kappa_ct_raw = nn.Parameter(torch.zeros(4))
        # positive BJ globals: a1, a2 >= 0.5 bohr, s8
        self.disp_raw = nn.Parameter(torch.tensor(
            [_softplus_inv(0.4), _softplus_inv(4.9 - 0.5), _softplus_inv(1.0)],
            dtype=torch.float64))
        self.trunk.to(trunk_dtype)
        for mod in (self.edge_head, self.charge_head, self.prop_head,
                    self.vec_readout, self.ten_readout):
            mod.to(trunk_dtype)

    # -- parameters ----------------------------------------------------------
    def gammas(self) -> tuple[Tensor, Tensor]:
        """(gamma_sub, gamma_val) per element; 1.0 and 1.85 at init."""
        g = 0.5 + torch.nn.functional.softplus(
            self.gamma_raw + torch.tensor([_softplus_inv(0.5), _softplus_inv(1.35)],
                                          dtype=self.gamma_raw.dtype,
                                          device=self.gamma_raw.device))
        return g[:, 0], g[:, 1]

    def dispersion_globals(self) -> dict:
        sp = torch.nn.functional.softplus(self.disp_raw)
        return {"a1": sp[0], "a2": 0.5 + sp[1], "s8": sp[2]}

    def element_parameters(self) -> list[nn.Parameter]:
        return [self.guess.weight, self.gamma_raw, self.disp_raw]

    # -- forward -------------------------------------------------------------
    def forward(self, batch: dict, return_matrix: bool = False):
        R_b, atoms = self._forward_atoms(batch)
        pairs = self._pairs(batch, R_b, atoms)
        n = batch["n"]
        pdim = batch["pair_dimer"]

        def per_dimer(x):
            return x.new_zeros(n).index_add(0, pdim, x)

        e_pol, pol = physics.polarization(
            R_b, batch["atom_dimer"], n, atoms["sqrt_alpha"], atoms["beta_pol"],
            pairs["field"], return_matrix=return_matrix)
        elst = per_dimer(pairs["elst"])
        exch = per_dimer(pairs["exch"])
        ct = per_dimer(pairs["ct"])
        disp = per_dimer(pairs["disp"])
        pred = torch.stack([elst, exch, e_pol + ct, disp], dim=1) * H2KCAL
        extras = {**{k: atoms[k] for k in ("q", "mu", "th", "alpha_iso", "nu",
                                           "n_act", "beta", "g_el", "cn")},
                  "e_pol": e_pol * H2KCAL, "e_ct": ct * H2KCAL,
                  "c6_pair": pairs["c6"], "exch_pair": pairs["exch"] * H2KCAL,
                  "disp_pair": pairs["disp"] * H2KCAL, "ct_pair": pairs["ct"] * H2KCAL,
                  "fragments_by_rule": atoms["fragments_by_rule"],
                  "induced_dipole": pol["induced_dipole"]}
        if return_matrix:
            extras["pol_matrix"] = pol["matrix"]
        return pred, extras

    def _forward_atoms(self, batch: dict):
        """(R in bohr, per-atom charges, moments and response properties)."""
        z = batch["z"].long()
        R = batch["R"]
        h, u, Q, geo = self.trunk(z, R.to(self.trunk_dtype), batch["edges"])
        atoms = self._charges(batch, z, h, geo)
        atoms.update(self._properties(z, h, u, Q, atoms, geo))
        return R.to(self.physics_dtype) * ANG2BOHR, atoms

    def _charges(self, batch, z, h, geo):
        pd = torch.float64
        src, dst = batch["edges"]
        frag, q_f, n_rule = ch.covalent_fragments(
            z, geo["r"].detach().double(), batch["edges"], batch["monomer"],
            batch["monomer_charge"], self.z_act.long(), self.r_cov,
            batch.get("fragment"), batch.get("fragment_charge"))
        n_frag = len(q_f)
        zact = self.z_act[z]
        same = (frag[src] == frag[dst]).to(pd)
        raw_g = self.edge_head(
            torch.cat([h[src], h[dst], geo["rbf"]], dim=1)).squeeze(-1).to(pd)
        g = G_CENTER + G_HALF * torch.tanh((raw_g - G_SHIFT) / G_HALF)
        w = geo["fc"].to(pd) * torch.exp(g) * same
        den = 1.0 + torch.zeros_like(zact).index_add(0, src, w)
        p = THETA_MAX * w / den[src]
        flow = zact[src] * p
        n_bond = zact - torch.zeros_like(zact).index_add(0, src, flow) \
            + torch.zeros_like(zact).index_add(0, dst, flow)
        raw = self.charge_head(h).to(pd)
        nu = RHO_NU * torch.tanh(raw[:, 0] / RHO_NU)
        s_pos = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 1])
        s_neg = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 2])
        zact_f = torch.zeros(n_frag, dtype=pd, device=z.device).index_add(0, frag, zact)
        n_act, mu_f = ch.fukui_populations(n_bond * torch.exp(nu), s_pos, s_neg,
                                           frag, zact_f - q_f)
        q = zact - n_act
        cn = torch.zeros_like(zact).index_add(0, src, geo["fc"].to(pd))
        return {"q": q, "n_act": n_act, "nu": nu, "frag": frag, "mu_f": mu_f,
                "cn": cn, "fragments_by_rule": n_rule}

    def _properties(self, z, h, u, Q, atoms, geo):
        pd = self.physics_dtype
        q, n_act, cn = atoms["q"].to(pd), atoms["n_act"].to(pd), atoms["cn"].to(pd)
        zs, zv = self.z_sub[z].to(pd), self.z_val[z].to(pd)
        has_sub = zs > 0
        n_val = torch.where(has_sub, phi_tau(n_act - zs) - phi_tau(-zs), n_act)
        n_sub = n_act - n_val
        f_sub = n_sub / zs.clamp_min(1.0)
        f_val = n_val / zv
        n_tot = self.n_core[z].to(pd) + n_act
        g_el = n_tot**2 / (n_tot**2 + EPS_EL**2)
        feats = torch.cat([h, torch.stack([q, n_sub, n_val, cn], 1).to(h.dtype)], 1)
        raw = self.prop_head(feats).to(pd)
        delta = {k: b * torch.tanh(raw[:, i] / b)
                 for i, (k, b) in enumerate(SCALAR_HEADS.items())}
        guess = self.guess(z).to(pd)
        gz = {k: guess[:, i] for i, k in enumerate(GUESS_NAMES)}
        vec = self.vec_readout(u).to(pd)   # (N, 3, n_vec)
        ten = self.ten_readout(Q).to(pd)   # (N, 3, 3, n_ten)
        mu = g_el[:, None] * vec[:, :, 0]
        th = g_el[:, None, None] * ten[..., 0]
        vecs = {k: norm_cap(vec[:, :, i + 1], b, (1,))
                for i, (k, b) in enumerate(VECTOR_HEADS.items())}
        tens = {k: norm_cap(ten[..., i + 1], b, (1, 2))
                for i, (k, b) in enumerate(TENSOR_HEADS.items())}
        g_sub, g_val = self.gammas()
        alpha_iso = (self.alpha_core[z].to(pd)
                     + self.d_alpha_sub[z].to(pd) * psi(f_sub, g_sub[z].to(pd))
                     + self.d_alpha_val[z].to(pd) * psi(f_val, g_val[z].to(pd))
                     ) * torch.exp(delta["alpha"])
        sqrt_alpha = (g_el * torch.sqrt(alpha_iso + EPS_ALPHA))[:, None, None] \
            * torch.linalg.matrix_exp(0.5 * tens["alpha"])
        ft = torch.tanh(f_val / 2.0) / math.tanh(0.5)
        e_cat, e_free = self.eta_cat[z].to(pd), self.eta_free[z].to(pd)
        eta = e_cat * (e_free / e_cat) ** ft * torch.exp(delta["eta"])

        def amp(name):
            return torch.exp(getattr(self, f"log_{name}0")[z].to(pd) + gz[name]
                             + delta[name])

        return {"mu": mu, "th": th, "g_el": g_el, "alpha_iso": alpha_iso,
                "sqrt_alpha": sqrt_alpha, "beta_pol": physics.beta_pol(alpha_iso),
                "eta": eta, "beta": amp("beta"),
                "A": g_el * amp("A"), "B": amp("B"),
                "D": g_el * amp("D"), "C": amp("C"),
                "eta_B": 0.8 + 0.2 * torch.sigmoid(delta["eta_B"]),
                "vec": vecs, "ten": tens}

    def _s_cut(self, r_bohr):
        if self.s_cut_window is None:
            return None
        on, off = (x * ANG2BOHR for x in self.s_cut_window)
        t = ((r_bohr - on) / (off - on)).clamp(0.0, 1.0)
        return 1 - 35 * t**4 + 84 * t**5 - 70 * t**6 + 20 * t**7

    def _pairs(self, batch, R_b, at):
        pd = self.physics_dtype
        z = batch["z"].long()
        i, j = batch["ai"], batch["aj"]
        r, rhat = gordon1.safe_unit(R_b[j] - R_b[i])
        sc = self._s_cut(r)
        zf = z.to(pd)

        def side(idx):
            return {"z": zf[idx], "nc": self.n_core[z[idx]].to(pd),
                    "bc": self.beta_core[z[idx]].to(pd), "n": at["n_act"][idx].to(pd),
                    "b": at["beta"][idx], "q": at["q"][idx].to(pd),
                    "mu": at["mu"][idx], "th": at["th"][idx]}

        si, sj = side(i), side(j)
        elst = gordon1.elst_pairs(r, rhat, si, sj)
        vec, ten = at["vec"], at["ten"]
        B_ij = at["B"][i] * torch.exp(physics.xi(vec["B"][i], ten["B"][i], rhat))
        B_ji = at["B"][j] * torch.exp(physics.xi(vec["B"][j], ten["B"][j], -rhat))
        pre_i = physics.xi(vec["pre"][i], ten["pre"][i], rhat)
        pre_j = physics.xi(vec["pre"][j], ten["pre"][j], -rhat)
        omega = physics.transverse_invariants(rhat, vec["ex"][i], ten["ex"][i],
                                              vec["ex"][j], ten["ex"][j])
        exch = physics.exchange_pairs(r, rhat, at["A"][i], at["A"][j], B_ij, B_ji,
                                      pre_i, pre_j, omega,
                                      torch.tanh(self.kappa_ex_raw).to(pd), sc)
        w_ct = torch.exp((omega * torch.tanh(self.kappa_ct_raw).to(pd)).sum(-1))
        d_ij = at["D"][i] * torch.exp(physics.xi(vec["D"][i], ten["D"][i], rhat))
        d_ji = at["D"][j] * torch.exp(physics.xi(vec["D"][j], ten["D"][j], -rhat))
        c_ij = at["C"][i] * torch.exp(physics.xi(vec["C"][i], ten["C"][i], rhat))
        c_ji = at["C"][j] * torch.exp(physics.xi(vec["C"][j], ten["C"][j], -rhat))
        gamma_sum = (d_ij * c_ji + d_ji * c_ij) * w_ct
        ct = physics.charge_transfer_pairs(
            r, exch, zf[i], zf[j], gamma_sum, at["eta_B"][i] * B_ij,
            at["eta_B"][j] * B_ji,
            eps_ct=1e-30 if pd == torch.float64 else 1e-15, s_cut=sc)
        dg = self.dispersion_globals()
        disp, c6 = physics.dispersion_pairs(
            r, at["alpha_iso"][i], at["alpha_iso"][j], at["eta"][i], at["eta"][j],
            self.r42[z[i]].to(pd), self.r42[z[j]].to(pd),
            self.alpha_free[z[i]].to(pd), self.alpha_free[z[j]].to(pd),
            dg["a1"].to(pd), dg["a2"].to(pd), dg["s8"].to(pd), s_cut=sc)
        # permanent fields on each side's polarizable sites from the partner
        bp = at["beta_pol"]
        f_on_i = gordon1.permanent_field(r, -rhat, bp[i], sj)
        f_on_j = gordon1.permanent_field(r, rhat, bp[j], si)
        if sc is not None:
            f_on_i, f_on_j = f_on_i * sc[:, None], f_on_j * sc[:, None]
        field = R_b.new_zeros(R_b.shape).index_add(0, i, f_on_i).index_add(0, j, f_on_j)
        return {"elst": elst, "exch": exch, "ct": ct, "disp": disp, "c6": c6,
                "field": field}


def _softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))
