"""FF-2: the best FF-1 (v5 MBIS + beta_def + short-range shell, one-mu
charges) with a forward built for ``torch.compile``.

FF-2 *is* FF-1's parameterisation: ``FF2Model`` subclasses
:class:`apnet_pt.ff1.FF1Model` with ``elst_beta_def`` and ``elst_sr_shell``
always on and ``g0_init = -2.5``, so an FF-1 state dict of that configuration
loads with no key mapping (and an FF-1 state dict without the two
electrostatics extensions loads too: their zero-initialised heads reproduce
the single-exponent model exactly, see :meth:`FF2Model.from_ff1`).  Only the
forward is new:

* charges: one chemical potential per monomer (``charges``), the FF-1
  ``fragment = monomer`` mode; FF-1's covalent-piece rule does not exist;
* every data-dependent index (polarization pairs, dense slot layout) is built
  by :func:`apnet_pt.ff2.batching.prepare` before the forward;
* per-head, per-shell and per-exponent-pair Python loops are batched into
  broadcast tensors (``physics``, ``kernels``); no ``.item()``, no
  ``nonzero``, no data-dependent branching, so ``torch.compile(fullgraph=True,
  dynamic=True)`` traces the whole forward and backward as one graph.

``FF2Model(batch)`` returns ``(pred, extras)`` with ``pred`` (n, 4) kcal/mol
``[elst, exch, ind, disp]`` exactly as FF-1.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..ff1.model import (BETA_DEF_BOUND, EPS_ALPHA, EPS_EL, G_CENTER, G_HALF,
                         GUESS_NAMES, H2KCAL, RHO_NU, SCALAR_HEADS,
                         SIGMA_MIN, SR_SHELL_CAP, TENSOR_HEADS, THETA_MAX,
                         VECTOR_HEADS, FF1Model, phi_tau, psi)
from ..ff1.trunk import ANG2BOHR, EPS_U
from . import physics
from .charges import N_ITERS, monomer_populations
from .kernels import safe_unit

G0_INIT = -2.5
OPTIONAL_FF1_KEYS = ("beta_def_guess.", "beta_def_head.", "sr_vec_readout.",
                     "sr_ten_readout.")


def norm_cap(x: Tensor, rho: Tensor, dims: tuple[int, ...]) -> Tensor:
    """FF-1's ``norm_cap`` with a per-head bound ``rho`` on the last axis."""
    n = torch.sqrt((x * x).sum(dims, keepdim=True) + EPS_U**2)
    return x * (rho * torch.tanh(n / rho) / n)


class FF2Model(FF1Model):
    """FF-2 force field (FF-1 v5 best configuration, compile-ready forward).

    Parameters
    ----------
    n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr
        Trunk hyperparameters (FF-1 v5: 128, 16, 3, 16, 5.0, 4.0).
    element_init : dict, optional
        Per-element starting tables (see ``ff1.model.default_element_init``).
    trunk_dtype, physics_dtype : torch.dtype
        Trunk precision (float32) and physics precision (float64); the
        one-mu charge solve always runs in float64.
    s_cut : (float, float), optional
        Septic condensed-phase switch window (Angstrom).
    g0_init : float
        Bond-flow logit prior; -2.5 (FF-1 v5).
    n_mu_iters : int
        Fixed Newton-bisection iterations of the one-mu solve.
    use_cholesky : bool
        Solve the SPD polarization system by Cholesky (else LU, as FF-1).
    """

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_layers: int = 3,
                 n_rbf: int = 16, r_cut: float = 5.0, rho_tr: float = 4.0,
                 element_init: dict | None = None,
                 trunk_dtype: torch.dtype = torch.float32,
                 physics_dtype: torch.dtype = torch.float64,
                 s_cut: tuple[float, float] | None = None,
                 g0_init: float = G0_INIT, n_mu_iters: int = N_ITERS,
                 use_cholesky: bool = True):
        super().__init__(n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr,
                         element_init, trunk_dtype, physics_dtype, s_cut,
                         g0_init, elst_beta_def=True, elst_sr_shell=True)
        self.n_mu_iters = int(n_mu_iters)
        self.use_cholesky = bool(use_cholesky)
        # beta_min R above which a pair's Slater kernels take their point-
        # multipole limit (physics.by_range); None keeps every pair damped
        self.point_x: float | None = None
        pd = physics_dtype
        for name, table in (("scalar_bound", SCALAR_HEADS),
                            ("vector_bound", VECTOR_HEADS),
                            ("tensor_bound", TENSOR_HEADS)):
            self.register_buffer(name, torch.tensor(list(table.values()), dtype=pd),
                                 persistent=False)
        # factor on E_CT's gradient (forward unchanged); a trainer schedules it
        # in place, so the compiled graph reads the current value
        self.register_buffer("ct_grad_scale", torch.tensor(1.0, dtype=pd),
                             persistent=False)

    # -- construction from FF-1 ----------------------------------------------
    @classmethod
    def from_ff1(cls, ff1: FF1Model, **options) -> "FF2Model":
        """FF-2 with ``ff1``'s hyperparameters and weights (no key mapping).

        An FF-1 model without ``elst_beta_def``/``elst_sr_shell`` loads with
        those heads at their zero-readout init, which reproduces it exactly.
        """
        t = ff1.trunk
        cfg = dict(n_feat=t.n_feat, n_chan=t.n_chan, n_layers=len(t.layers),
                   n_rbf=t.rbf.centers.numel(), r_cut=t.r_cut,
                   rho_tr=t.layers[0].rho, trunk_dtype=ff1.trunk_dtype,
                   physics_dtype=ff1.physics_dtype, s_cut=ff1.s_cut_window,
                   g0_init=ff1.g0_init)
        model = cls(**{**cfg, **options})
        model.load_ff1_state_dict(ff1.state_dict(), keep_dtype=True)
        return model.to(next(ff1.parameters()).device)

    def load_ff1_state_dict(self, state: dict, keep_dtype: bool = False):
        """Load an FF-1 state dict; only the optional electrostatics heads may
        be missing (they keep their zero-readout init).  ``keep_dtype`` takes
        each tensor's dtype from ``state`` (e.g. an FF-1 model cast to float64
        throughout) instead of casting into this model's parameters."""
        if keep_dtype:
            state = {k: v.detach().clone() for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False,
                                                   assign=keep_dtype)
        bad = [k for k in missing if not k.startswith(OPTIONAL_FF1_KEYS)]
        if bad or unexpected:
            raise ValueError(f"not an FF-1 state dict: missing {bad}, "
                             f"unexpected {unexpected}")
        return missing

    def config(self) -> dict:
        return {**super().config(), "n_mu_iters": self.n_mu_iters,
                "use_cholesky": self.use_cholesky}

    def ct_parameter_slices(self) -> list[tuple[nn.Parameter, int, list[int]]]:
        """``(parameter, dim, indices)`` of every entry that reaches the energy
        only through E_CT: the D/C element offsets, the D/C/eta_B rows of the
        scalar readout, the D/C rows of the vector and tensor readouts, and the
        CT anisotropy weights.  The trunk and hidden layers are shared."""
        g = [GUESS_NAMES.index(k) for k in ("D", "C")]
        s = [list(SCALAR_HEADS).index(k) for k in ("D", "C", "eta_B")]
        v = [1 + list(VECTOR_HEADS).index(k) for k in ("D", "C")]
        t = [1 + list(TENSOR_HEADS).index(k) for k in ("D", "C")]
        last = self.prop_head[-1]
        return [(self.guess.weight, 1, g), (last.weight, 0, s), (last.bias, 0, s),
                (self.vec_readout.weight, 0, v), (self.ten_readout.weight, 0, t),
                (self.kappa_ct_raw, 0, list(range(self.kappa_ct_raw.numel())))]

    # -- forward -------------------------------------------------------------
    def forward(self, batch: dict, return_matrix: bool = False):
        """``batch`` from :func:`apnet_pt.ff2.batching.prepare`."""
        if return_matrix:
            raise ValueError("FF-2 never forms a differentiable polarization "
                             "matrix; use FF-1 for return_matrix")
        R_b, at = self._forward_atoms(batch)
        n = batch["n_dimer"].shape[0]
        pr = self._pairs(batch, R_b, at)
        e_pol, induced = physics.polarization(R_b, at["sqrt_alpha"], at["beta_pol"],
                                              pr["field"], batch, self.use_cholesky,
                                              self.point_x)
        pdim = batch["pair_dimer"]
        sums = R_b.new_zeros((n, 4)).index_add(
            0, pdim, torch.stack([pr["elst"], pr["exch"], pr["ct"], pr["disp"]], 1))
        elst, exch, ct, disp = sums.unbind(1)
        pred = torch.stack([elst, exch, e_pol + ct, disp], dim=1) * H2KCAL
        extras = {**{k: at[k] for k in ("q", "mu", "th", "alpha_iso", "nu", "n_act",
                                         "beta", "g_el", "cn", "beta_def", "d_sr",
                                         "Q_sr", "mu_residual")},
                  "e_pol": e_pol * H2KCAL, "e_ct": ct * H2KCAL,
                  "c6_pair": pr["c6"], "exch_pair": pr["exch"] * H2KCAL,
                  "disp_pair": pr["disp"] * H2KCAL, "ct_pair": pr["ct"] * H2KCAL,
                  "fragments_by_rule": 0, "induced_dipole": induced}
        return pred, extras

    def forward_monomers(self, batch: dict):
        """Monomer-only forward ``(R in bohr, atoms)``; ``batch`` needs ``z``,
        ``R``, ``edges``, ``monomer`` and ``monomer_charge`` only."""
        return self._forward_atoms(batch)

    def _forward_atoms(self, batch: dict):
        z = batch["z"].long()
        R = batch["R"]
        h, u, Q, geo = self.trunk(z, R.to(self.trunk_dtype), batch["edges"])
        atoms = self._charges(batch, z, h, geo)
        atoms.update(self._properties(z, h, u, Q, atoms, geo))
        return R.to(self.physics_dtype) * ANG2BOHR, atoms

    def _charges(self, batch, z, h, geo):
        pd = torch.float64
        src, dst = batch["edges"]
        mon = batch["monomer"]
        q_mon = batch["monomer_charge"].to(pd)
        zact = self.z_act[z]
        raw_g = self.edge_head(
            torch.cat([h[src], h[dst], geo["rbf"]], dim=1)).squeeze(-1).to(pd)
        g = G_CENTER + G_HALF * torch.tanh((raw_g - self.g_shift) / G_HALF)
        w = geo["fc"].to(pd) * torch.exp(g)
        den = 1.0 + torch.zeros_like(zact).index_add(0, src, w)
        flow = zact[src] * (THETA_MAX * w / den[src])
        n_bond = zact - torch.zeros_like(zact).index_add(0, src, flow) \
            + torch.zeros_like(zact).index_add(0, dst, flow)
        raw = self.charge_head(h).to(pd)
        nu = RHO_NU * torch.tanh(raw[:, 0] / RHO_NU)
        s_pos = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 1])
        s_neg = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 2])
        target = q_mon.new_zeros(q_mon.shape).index_add(0, mon, zact) - q_mon
        n_act, mu_m, resid = monomer_populations(
            n_bond * torch.exp(nu), s_pos, s_neg, mon, target, self.n_mu_iters)
        cn = torch.zeros_like(zact).index_add(0, src, geo["fc"].to(pd))
        return {"q": zact - n_act, "n_act": n_act, "nu": nu, "mu_m": mu_m,
                "mu_residual": resid, "cn": cn}

    def _properties(self, z, h, u, Q, atoms, geo):
        pd = self.physics_dtype
        q, n_act, cn = atoms["q"].to(pd), atoms["n_act"].to(pd), atoms["cn"].to(pd)
        zs, zv = self.z_sub[z].to(pd), self.z_val[z].to(pd)
        n_val = torch.where(zs > 0, phi_tau(n_act - zs) - phi_tau(-zs), n_act)
        n_sub = n_act - n_val
        f_sub, f_val = n_sub / zs.clamp_min(1.0), n_val / zv
        n_tot = self.n_core[z].to(pd) + n_act
        g_el = n_tot**2 / (n_tot**2 + EPS_EL**2)
        feats = torch.cat([h, torch.stack([q, n_sub, n_val, cn], 1).to(h.dtype)], 1)
        raw = self.prop_head(feats).to(pd)
        sb = self.scalar_bound.to(pd)
        delta = dict(zip(SCALAR_HEADS, (sb * torch.tanh(raw / sb)).unbind(1)))
        gz = dict(zip(GUESS_NAMES, self.guess(z).to(pd).unbind(1)))
        vec = self.vec_readout(u).to(pd)   # (N, 3, 1 + n_vector_heads)
        ten = self.ten_readout(Q).to(pd)   # (N, 3, 3, 1 + n_tensor_heads)
        mu = g_el[:, None] * vec[:, :, 0]
        th = g_el[:, None, None] * ten[..., 0]
        vecs = norm_cap(vec[:, :, 1:], self.vector_bound.to(pd), (1,))
        tens = norm_cap(ten[..., 1:], self.tensor_bound.to(pd), (1, 2))
        g_sub, g_val = self.gammas()
        alpha_iso = (self.alpha_core[z].to(pd)
                     + self.d_alpha_sub[z].to(pd) * psi(f_sub, g_sub[z].to(pd))
                     + self.d_alpha_val[z].to(pd) * psi(f_val, g_val[z].to(pd))
                     ) * torch.exp(delta["alpha"])
        sqrt_alpha = (g_el * torch.sqrt(alpha_iso + EPS_ALPHA))[:, None, None] \
            * physics.expm_sym(0.5 * tens[..., -1])
        ft = torch.tanh(f_val / 2.0) / math.tanh(0.5)
        e_cat, e_free = self.eta_cat[z].to(pd), self.eta_free[z].to(pd)
        eta = e_cat * (e_free / e_cat) ** ft * torch.exp(delta["eta"])

        def amp(name):
            return torch.exp(getattr(self, f"log_{name}0")[z].to(pd) + gz[name]
                             + delta[name])

        beta = amp("beta")
        raw_def = self.beta_def_head(feats).to(pd)[:, 0]
        d_def = BETA_DEF_BOUND * torch.tanh(raw_def / BETA_DEF_BOUND)
        beta_def = beta * torch.exp(self.beta_def_guess(z).to(pd)[:, 0] + d_def)
        cap = torch.tensor(SR_SHELL_CAP, dtype=pd, device=raw.device)
        d_sr = g_el[:, None] * norm_cap(self.sr_vec_readout(u).to(pd), cap,
                                        (1,))[..., 0]
        Q_sr = g_el[:, None, None] * norm_cap(self.sr_ten_readout(Q).to(pd), cap,
                                              (1, 2))[..., 0]
        return {"mu": mu, "th": th, "g_el": g_el, "alpha_iso": alpha_iso,
                "sqrt_alpha": sqrt_alpha, "beta_pol": physics.beta_pol(alpha_iso),
                "eta": eta, "beta": beta, "A": g_el * amp("A"), "B": amp("B"),
                "D": g_el * amp("D"), "C": amp("C"),
                "eta_B": 0.8 + 0.2 * torch.sigmoid(delta["eta_B"]),
                "vecs": vecs, "tens": tens, "beta_def": beta_def,
                "d_sr": d_sr, "Q_sr": Q_sr,
                # the two multipole shells (FF-1 multipole_shells)
                "shell_M": torch.stack([mu + d_sr, -d_sr], 1),
                "shell_T": torch.stack([th + Q_sr, -Q_sr], 1),
                "shell_b": torch.stack([beta_def, beta], 1)}

    def _side(self, idx, zf, at):
        z, pd = zf.long()[idx], zf.dtype
        return {"z": zf[idx], "nc": self.n_core[z].to(pd),
                "bc": self.beta_core[z].to(pd), "n": at["n_act"][idx].to(pd),
                "b": at["beta"][idx], "q": at["q"][idx].to(pd),
                "M": at["shell_M"][idx], "T": at["shell_T"][idx],
                "bs": at["shell_b"][idx]}

    @staticmethod
    def _beta_min(side):
        """Smallest Slater exponent a pair side's elst/field kernels use."""
        return torch.minimum(torch.minimum(side["b"], side["bc"]),
                             side["bs"].min(-1).values)

    def _pairs(self, batch, R_b, at):
        pd = self.physics_dtype
        z = batch["z"].long()
        zf = z.to(pd)
        i, j = batch["ai"], batch["aj"]
        r, rhat = safe_unit(R_b[j] - R_b[i])
        sc = self._s_cut(r)
        si, sj = self._side(i, zf, at), self._side(j, zf, at)
        elst = physics.by_range(physics.elst_pairs, r * torch.minimum(
            self._beta_min(si), self._beta_min(sj)), self.point_x, r, rhat, si, sj)
        # heads: vector (pre, B, ex, D, C), tensor (pre, B, ex, D, C, alpha)
        vecs, tens = at["vecs"], at["tens"]
        vi, vj, ti, tj = vecs[i], vecs[j], tens[i], tens[j]
        rq = rhat[:, :, None, None] * rhat[:, None, :, None]
        quad_i, quad_j = (ti[..., :5] * rq).sum((1, 2)), (tj[..., :5] * rq).sum((1, 2))
        lin_i = (vi * rhat[:, :, None]).sum(1)
        lin_j = (vj * rhat[:, :, None]).sum(1)
        xi_i, xi_j = lin_i + quad_i, quad_j - lin_j            # (P, 5)
        pre_i, b_i, _, d_i, c_i = xi_i.unbind(1)
        pre_j, b_j, _, d_j, c_j = xi_j.unbind(1)
        B_ij = at["B"][i] * torch.exp(b_i)
        B_ji = at["B"][j] * torch.exp(b_j)
        omega = physics.transverse_invariants(rhat, vi[:, :, 2], ti[..., 2],
                                              vj[:, :, 2], tj[..., 2])
        exch = physics.exchange_pairs(r, rhat, at["A"][i], at["A"][j], B_ij, B_ji,
                                      pre_i, pre_j, omega,
                                      torch.tanh(self.kappa_ex_raw).to(pd), sc)
        w_ct = torch.exp((omega * torch.tanh(self.kappa_ct_raw).to(pd)).sum(-1))
        gamma_sum = (at["D"][i] * torch.exp(d_i) * at["C"][j] * torch.exp(c_j)
                     + at["D"][j] * torch.exp(d_j) * at["C"][i] * torch.exp(c_i)) * w_ct
        ct = physics.charge_transfer_pairs(
            r, exch, zf[i], zf[j], gamma_sum, at["eta_B"][i] * B_ij,
            at["eta_B"][j] * B_ji,
            eps_ct=1e-30 if pd == torch.float64 else 1e-15, s_cut=sc)
        s = self.ct_grad_scale.to(pd)
        ct = s * ct + (1.0 - s) * ct.detach()  # exactly ct at s = 1
        dg = self.dispersion_globals()
        disp, c6 = physics.dispersion_pairs(
            r, at["alpha_iso"][i], at["alpha_iso"][j], at["eta"][i], at["eta"][j],
            self.r42[z[i]].to(pd), self.r42[z[j]].to(pd),
            self.alpha_free[z[i]].to(pd), self.alpha_free[z[j]].to(pd),
            dg["a1"].to(pd), dg["a2"].to(pd), dg["s8"].to(pd), s_cut=sc)
        # permanent fields: probes (i, j) <- sources (j, i), both in one call
        probe, source = torch.cat([i, j]), torch.cat([j, i])
        src, r2, bp = self._side(source, zf, at), torch.cat([r, r]), at["beta_pol"][probe]
        f = physics.by_range(physics.permanent_field,
                             r2 * torch.minimum(bp, self._beta_min(src)), self.point_x,
                             r2, torch.cat([-rhat, rhat]), bp, src)
        if sc is not None:
            f = f * torch.cat([sc, sc])[:, None]
        field = R_b.new_zeros(R_b.shape).index_add(0, probe, f)
        return {"elst": elst, "exch": exch, "ct": ct, "disp": disp, "c6": c6,
                "field": field}


class CompiledFF2(nn.Module):
    """``torch.compile(model, fullgraph=True, dynamic=True)`` traced with

    * duck shaping off, so a chunk whose dimer or atom count happens to equal
      a model constant (3 coordinates, 4 invariants, ...) is not specialized;
    * ``backed_size_oblivious``, so a one-dimer chunk (n == 1, slots m == N
      atoms) does not specialize sizes to 1 or to each other -- on real store
      batches every such chunk recompiled until the recompile limit;
    * ``opt_einsum`` off: with it installed, ``torch.einsum`` with three or
      more operands (FF-1's trunk invariants) picks its contraction path from
      the concrete sizes, which guards the atom count to its traced value and
      recompiles every new chunk.  Without it the operands contract left to
      right.

    One graph then serves every chunk shape.  Parameters stay ``model``'s."""

    def __init__(self, model: FF2Model, **options):
        super().__init__()
        self.model = model
        self._compiled = torch.compile(model, **{"fullgraph": True, "dynamic": True,
                                                 **options})

    def forward(self, batch: dict):
        from torch.fx.experimental import _config as fx_config
        with fx_config.patch(use_duck_shape=False, backed_size_oblivious=True), \
                torch.backends.opt_einsum.flags(enabled=False):
            return self._compiled(batch)


def compile_ff2(model: FF2Model, **options) -> CompiledFF2:
    """The compiled FF-2 forward (see :class:`CompiledFF2`)."""
    return CompiledFF2(model, **options)
