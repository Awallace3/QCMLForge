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
from .so3lr_backbone import BACKBONES, SO3LRFeatureTrunk
from .trunk import FF3MonomerTrunk, identity_init_layer

G0_INIT = -2.5
OPTIONAL_FF1_KEYS = ("beta_def_guess.", "beta_def_head.", "sr_vec_readout.",
                     "sr_ten_readout.", "a_pol_field_raw")
# FF3Layer modules that FF1Layer lacks (identity at init)
GATE_MODULES = ("wq", "wk", "filt1", "filt2", "w_ev")
POL_FIELD_MODES = ("charge_penetration", "net_multipole")
CHARGE_PRIORS = ("bond_flow", "so3lr_qeq")
QEQ_N_FLOOR = 0.05
A_POL_FIELD_MIN = 0.15
A_POL_FIELD_MAX = 2.50
A_POL_FIELD_INIT = 0.70


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
    sr_shell, charge_transfer, transverse_invariants, alpha_anisotropy : bool
        Ablation switches (default on = FF-2): the short-range multipole
        shell ``d_sr``/``Q_sr``; the CT energy; the SO(2) invariants omega
        in exchange and CT (``kappa . omega``); the anisotropic factor
        ``exp(tau_alpha / 2)`` of sqrt(alpha).  Off removes the term from
        the forward; its parameters stay in the state dict (unused).
    element_alpha_eta : bool
        Ablation (default off): ``alpha_iso = alpha_free[z] exp(delta_alpha)``
        and ``eta = eta_free[z] exp(delta_eta)`` replace the population
        (core/sub/valence psi) and cation-to-free interpolation models.
    atomic_dipoles, atomic_quadrupoles : bool
        Ablation switches (default on): the atomic dipole ``mu`` and the
        traceless quadrupole ``Theta``.  Off sets that moment to zero in the
        electrostatics and in the permanent field, together with its
        short-range partner (``d_sr`` / ``Q_sr``) when ``sr_shell`` is on;
        the charges are unchanged.
    charge_embedding, attention, euclidean_variables : bool
        FF-3 backbone arms (default off; :mod:`apnet_pt.ff2.trunk`): the
        SO3LR total-charge embedding (E), the So3krates attention gate on
        the trunk messages (A) and the l = 1..4 Euclidean-variable
        descriptor (A-ev).  Each starts as an exact identity, and the
        trunk's FF-1 weights are those the default model would draw.
    n_identity_layers : int
        FF-3 capacity arm (CAP): the last ``n_identity_layers`` of the
        ``n_layers`` trunk layers start with every update path at zero and
        cap their increments, ``u + G_rho(du)``, instead of their state, so
        they are an exact identity at init and stay bounded (``2 rho``).
    so3lr_backbone : str
        FF-3 frozen-backbone arms (default ``""`` = FF-1's trunk;
        :mod:`apnet_pt.ff2.so3lr_backbone`): ``"tl-linear"`` (C) or
        ``"tl-adapter"`` (D) replace the trunk by maps of cached SO3LR-T-L
        features (batch keys ``so3lr_x``, ``so3lr_ev``), D with one FF1Layer
        adapter.  ``n_layers`` is then unused, and an FF-1 state dict does not
        load (the trunk keys differ); the heads are unchanged.
    """

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_layers: int = 3,
                 n_rbf: int = 16, r_cut: float = 5.0, rho_tr: float = 4.0,
                 element_init: dict | None = None,
                 trunk_dtype: torch.dtype = torch.float32,
                 physics_dtype: torch.dtype = torch.float64,
                 s_cut: tuple[float, float] | None = None,
                 g0_init: float = G0_INIT, n_mu_iters: int = N_ITERS,
                 use_cholesky: bool = True, sr_shell: bool = True,
                 charge_transfer: bool = True, transverse_invariants: bool = True,
                 alpha_anisotropy: bool = True, element_alpha_eta: bool = False,
                 atomic_dipoles: bool = True, atomic_quadrupoles: bool = True,
                 charge_embedding: bool = False, attention: bool = False,
                 euclidean_variables: bool = False, n_identity_layers: int = 0,
                 so3lr_backbone: str = "",
                 pol_field: str = "charge_penetration",
                 learnable_a_pol_field: bool = False,
                 a_pol_field_init: float | None = None,
                 covalent_bond_flow: bool = False,
                 charge_prior: str = "bond_flow",
                 rho_nu: float = RHO_NU):
        super().__init__(n_feat, n_chan, n_layers, n_rbf, r_cut, rho_tr,
                         element_init, trunk_dtype, physics_dtype, s_cut,
                         g0_init, elst_beta_def=True, elst_sr_shell=True)
        if pol_field not in POL_FIELD_MODES:
            raise ValueError(f"pol_field {pol_field!r} not in {POL_FIELD_MODES}")
        if charge_prior not in CHARGE_PRIORS:
            raise ValueError(f"charge_prior {charge_prior!r} not in {CHARGE_PRIORS}")
        if charge_prior == "so3lr_qeq" and not so3lr_backbone:
            raise ValueError("charge_prior='so3lr_qeq' requires so3lr_backbone")
        if learnable_a_pol_field and pol_field != "net_multipole":
            raise ValueError("learnable_a_pol_field requires pol_field='net_multipole'")
        if pol_field != "net_multipole" and a_pol_field_init is not None:
            raise ValueError("a_pol_field_init requires pol_field='net_multipole'")
        self.n_mu_iters = int(n_mu_iters)
        self.use_cholesky = bool(use_cholesky)
        self.sr_shell = bool(sr_shell)
        self.charge_transfer = bool(charge_transfer)
        self.transverse_invariants = bool(transverse_invariants)
        self.alpha_anisotropy = bool(alpha_anisotropy)
        self.element_alpha_eta = bool(element_alpha_eta)
        self.atomic_dipoles = bool(atomic_dipoles)
        self.atomic_quadrupoles = bool(atomic_quadrupoles)
        self.covalent_bond_flow = bool(covalent_bond_flow)
        self.charge_prior = str(charge_prior)
        self.rho_nu = float(rho_nu)
        self.pol_field = pol_field
        self.learnable_a_pol_field = bool(learnable_a_pol_field)
        if self.pol_field == "net_multipole":
            init_val = (A_POL_FIELD_INIT if self.learnable_a_pol_field else physics.A_POL) \
                if a_pol_field_init is None else float(a_pol_field_init)
            if self.learnable_a_pol_field:
                if not (A_POL_FIELD_MIN < init_val < A_POL_FIELD_MAX):
                    raise ValueError(
                        f"a_pol_field_init {init_val} must lie strictly in "
                        f"({A_POL_FIELD_MIN}, {A_POL_FIELD_MAX})")
                p0 = (init_val - A_POL_FIELD_MIN) / (A_POL_FIELD_MAX - A_POL_FIELD_MIN)
                logit0 = math.log(p0 / (1.0 - p0))
                self.a_pol_field_raw = nn.Parameter(
                    torch.tensor(logit0, dtype=torch.float64))
            elif init_val <= 0.0:
                raise ValueError(f"a_pol_field_init must be > 0, got {init_val}")
            self.a_pol_field_init: float | None = init_val
        else:
            self.a_pol_field_init = None
        # beta_min R above which a pair's Slater kernels take their point-
        # multipole limit (physics.by_range); None keeps every pair damped
        self.point_x: float | None = None
        # Slater kernels within point_x: "exact" (series), "table" (cubic
        # Hermite in x, kernels.two_center_table) or "table2d" (bicubic
        # Hermite in (beta_min R, beta_min / beta_max) replacing the whole
        # quadrature, kernels.two_center_table2d)
        self.kernel_mode: str = "exact"
        # polarization by block-Jacobi CG with this many steps (None: dense
        # Cholesky / LU per use_cholesky)
        self.pol_cg_iters: int | None = None
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
        self._build_backbone(charge_embedding, attention, euclidean_variables,
                             n_identity_layers, so3lr_backbone)

    def _build_backbone(self, charge_embedding, attention, euclidean_variables,
                        n_identity_layers, so3lr_backbone=""):
        """The FF-3 trunk options; all off leaves FF-1's trunk untouched."""
        self.so3lr_backbone = str(so3lr_backbone or "")
        if self.so3lr_backbone:
            if self.so3lr_backbone not in BACKBONES:
                raise ValueError(f"so3lr_backbone {so3lr_backbone!r} not in {BACKBONES}")
            if charge_embedding or attention or euclidean_variables or n_identity_layers:
                raise ValueError("so3lr_backbone replaces the FF-1 trunk; it does not "
                                 "combine with the FF-3 trunk options")
            base = self.trunk
            self.charge_embedding = self.attention = self.euclidean_variables = False
            self.n_identity_layers = 0
            self.trunk = SO3LRFeatureTrunk(
                base.n_feat, base.n_chan, base.rbf.centers.numel(), base.r_cut,
                base.layers[0].rho, adapter=self.so3lr_backbone == "tl-adapter"
            ).to(self.trunk_dtype)
            return
        self.charge_embedding = bool(charge_embedding)
        self.attention = bool(attention)
        self.euclidean_variables = bool(euclidean_variables)
        self.n_identity_layers = int(n_identity_layers)
        base = self.trunk
        if not 0 <= self.n_identity_layers <= len(base.layers):
            raise ValueError(f"n_identity_layers {n_identity_layers} not in "
                             f"[0, n_layers = {len(base.layers)}]")
        if self.charge_embedding or self.attention or self.euclidean_variables:
            trunk = FF3MonomerTrunk(
                base.n_feat, base.n_chan, len(base.layers), base.rbf.centers.numel(),
                base.r_cut, base.layers[0].rho, base.embed.num_embeddings,
                charge_embedding, attention, euclidean_variables)
            trunk.to(self.trunk_dtype)
            missing, unexpected = trunk.load_state_dict(base.state_dict(), strict=False)
            assert not unexpected and all(
                ("trunk." + k) in self.identity_init_keys(trunk) for k in missing)
            self.trunk = trunk
        for layer in self.trunk.layers[len(self.trunk.layers) - self.n_identity_layers:]:
            identity_init_layer(layer)

    def identity_init_keys(self, trunk: nn.Module | None = None) -> list[str]:
        """State-dict keys that start at identity init and that an FF-1 (or
        MBIS) state dict of ``n_layers - n_identity_layers`` layers lacks: the
        charge embedding, every layer's gate/EV modules and the identity
        layers."""
        trunk = self.trunk if trunk is None else trunk
        if self.so3lr_backbone:
            return []
        n_base = len(trunk.layers) - self.n_identity_layers
        out = []
        for key in trunk.state_dict():
            parts = key.split(".")
            if parts[0] == "charge_embed" or (
                    parts[0] == "layers" and (int(parts[1]) >= n_base
                                              or parts[2] in GATE_MODULES)):
                out.append("trunk." + key)
        return out

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
        """Load an FF-1 state dict; only the optional electrostatics heads and
        the FF-3 identity-init keys (:meth:`identity_init_keys`) may be
        missing (they keep their init).  ``keep_dtype`` takes
        each tensor's dtype from ``state`` (e.g. an FF-1 model cast to float64
        throughout) instead of casting into this model's parameters."""
        if keep_dtype:
            state = {k: v.detach().clone() for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False,
                                                   assign=keep_dtype)
        new = set(self.identity_init_keys())
        bad = [k for k in missing if not k.startswith(OPTIONAL_FF1_KEYS)
               and k not in new]
        if bad or unexpected:
            raise ValueError(f"not an FF-1 state dict: missing {bad}, "
                             f"unexpected {unexpected}")
        return missing

    def config(self) -> dict:
        out = {**super().config(), "n_mu_iters": self.n_mu_iters,
               "use_cholesky": self.use_cholesky, **self.ablations(),
               **self.backbone(), "pol_field": self.pol_field,
               "learnable_a_pol_field": self.learnable_a_pol_field}
        if self.a_pol_field_init is not None:
            out["a_pol_field_init"] = self.a_pol_field_init
        if self.covalent_bond_flow:
            out["covalent_bond_flow"] = True
        if self.charge_prior != "bond_flow":
            out["charge_prior"] = self.charge_prior
        if self.rho_nu != RHO_NU:
            out["rho_nu"] = self.rho_nu
        return out

    def a_pol_field(self) -> Tensor | None:
        """Direct permanent-field damping constant ``a_d`` (matched to Thole
        damping; mutual polarization keeps ``A_POL = 0.385`` fixed for SPD)."""
        if self.pol_field != "net_multipole":
            return None
        if self.learnable_a_pol_field:
            return (A_POL_FIELD_MIN
                    + (A_POL_FIELD_MAX - A_POL_FIELD_MIN)
                    * torch.sigmoid(self.a_pol_field_raw)).to(self.physics_dtype)
        assert self.a_pol_field_init is not None
        return torch.tensor(self.a_pol_field_init, dtype=self.physics_dtype,
                            device=self.disp_raw.device)

    def pol_field_scale(self) -> Tensor | float:
        """Scalar multiplier ``c_dir = (a_pol_field / A_POL)^(1/3)`` applied to
        ``beta_pol`` inside ``permanent_field_net``."""
        if self.learnable_a_pol_field:
            a = self.a_pol_field()
            assert a is not None
            return (a / physics.A_POL) ** (1.0 / 3.0)
        assert self.a_pol_field_init is not None
        return (self.a_pol_field_init / physics.A_POL) ** (1.0 / 3.0)

    def element_parameters(self) -> list[nn.Parameter]:
        out = super().element_parameters()
        if self.learnable_a_pol_field:
            out.append(self.a_pol_field_raw)
        return out

    def backbone(self) -> dict:
        """The FF-3 trunk options (all off = FF-2's trunk)."""
        return {"charge_embedding": self.charge_embedding,
                "attention": self.attention,
                "euclidean_variables": self.euclidean_variables,
                "n_identity_layers": self.n_identity_layers,
                "so3lr_backbone": self.so3lr_backbone}

    def ablations(self) -> dict:
        """The ablation switches (all at their FF-2 default = no ablation)."""
        return {"sr_shell": self.sr_shell, "charge_transfer": self.charge_transfer,
                "transverse_invariants": self.transverse_invariants,
                "alpha_anisotropy": self.alpha_anisotropy,
                "element_alpha_eta": self.element_alpha_eta,
                "atomic_dipoles": self.atomic_dipoles,
                "atomic_quadrupoles": self.atomic_quadrupoles}

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
                                              self.point_x, self.kernel_mode,
                                              self.pol_cg_iters)
        pdim = batch["pair_dimer"]
        sums = R_b.new_zeros((n, 4)).index_add(
            0, pdim, torch.stack([pr["elst"], pr["exch"], pr["ct"], pr["disp"]], 1))
        elst, exch, ct, disp = sums.unbind(1)
        pred = torch.stack([elst, exch, e_pol + ct, disp], dim=1) * H2KCAL
        extras = {**{k: at[k] for k in ("q", "mu", "th", "alpha_iso", "nu", "n_act",
                                         "beta", "g_el", "cn", "beta_def", "d_sr",
                                         "Q_sr", "mu_residual", "n0", "s_pos", "s_neg")},
                  "e_pol": e_pol * H2KCAL, "e_ct": ct * H2KCAL,
                  "c6_pair": pr["c6"], "exch_pair": pr["exch"] * H2KCAL,
                  "disp_pair": pr["disp"] * H2KCAL, "ct_pair": pr["ct"] * H2KCAL,
                  "fragments_by_rule": 0, "induced_dipole": induced}
        return pred, extras

    def forward_monomers(self, batch: dict):
        """Monomer-only forward ``(R in bohr, atoms)``; ``batch`` needs ``z``,
        ``R``, ``edges``, ``monomer`` and ``monomer_charge`` only (plus
        ``so3lr_x``/``so3lr_ev`` with ``so3lr_backbone``)."""
        return self._forward_atoms(batch)

    def _forward_atoms(self, batch: dict):
        z = batch["z"].long()
        R = batch["R"]
        if self.so3lr_backbone:
            h, u, Q, geo = self.trunk(z, R.to(self.trunk_dtype), batch["edges"],
                                      batch["so3lr_x"], batch["so3lr_ev"])
        elif self.charge_embedding:
            h, u, Q, geo = self.trunk(z, R.to(self.trunk_dtype), batch["edges"],
                                      batch["monomer"], batch["monomer_charge"])
        else:
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
        if self.charge_prior == "so3lr_qeq":
            # Start from SO3LR-T-L QEq populations; edge_head is a bounded,
            # strictly conservative residual flux that is identically 0 when
            # edge_head[-1] is zeroed (so step 0 matches q_so3lr to round-off)
            # and guarantees n_bond >= 0.60 * n_start > 0 without clamping.
            n_start = (zact - batch["so3lr_q"].to(pd)).clamp_min(QEQ_N_FLOOR)
            mask = geo["fc"].to(pd)
            if self.covalent_bond_flow:
                same = batch.get("edge_same_frag")
                if same is None:
                    from ..ff1 import charges as _ff1_ch
                    bonded = (geo["r"].detach().to(pd)
                              < 1.25 * (self.r_cov[z[src]] + self.r_cov[z[dst]]))
                    frag = _ff1_ch.connected_components(len(z), batch["edges"][:, bonded])
                    same = frag[src] == frag[dst]
                mask = mask * same.to(pd)
            deg = torch.zeros_like(zact).index_add(0, src, mask).clamp_min(1.0)
            cap = torch.minimum(n_start[src], n_start[dst])
            scale = torch.minimum(1.0 / deg[src], 1.0 / deg[dst])
            flow = 0.20 * mask * torch.tanh(raw_g) * cap * scale
            n_bond = n_start - torch.zeros_like(zact).index_add(0, src, flow) \
                + torch.zeros_like(zact).index_add(0, dst, flow)
        else:
            g = G_CENTER + G_HALF * torch.tanh((raw_g - self.g_shift) / G_HALF)
            w = geo["fc"].to(pd) * torch.exp(g)
            if self.covalent_bond_flow:
                same = batch.get("edge_same_frag")
                if same is None:
                    from ..ff1 import charges as _ff1_ch
                    bonded = (geo["r"].detach().to(pd)
                              < 1.25 * (self.r_cov[z[src]] + self.r_cov[z[dst]]))
                    frag = _ff1_ch.connected_components(len(z), batch["edges"][:, bonded])
                    same = frag[src] == frag[dst]
                w = w * same.to(pd)
            den = 1.0 + torch.zeros_like(zact).index_add(0, src, w)
            flow = zact[src] * (THETA_MAX * w / den[src])
            n_bond = zact - torch.zeros_like(zact).index_add(0, src, flow) \
                + torch.zeros_like(zact).index_add(0, dst, flow)
        raw = self.charge_head(h).to(pd)
        nu = self.rho_nu * torch.tanh(raw[:, 0] / self.rho_nu)
        s_pos = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 1])
        s_neg = SIGMA_MIN + torch.nn.functional.softplus(raw[:, 2])
        target = q_mon.new_zeros(q_mon.shape).index_add(0, mon, zact) - q_mon
        n0 = n_bond * torch.exp(nu)
        n_act, mu_m, resid = monomer_populations(
            n0, s_pos, s_neg, mon, target, self.n_mu_iters)
        cn = torch.zeros_like(zact).index_add(0, src, geo["fc"].to(pd))
        return {"q": zact - n_act, "n_act": n_act, "nu": nu, "mu_m": mu_m,
                "mu_residual": resid, "cn": cn, "n0": n0, "s_pos": s_pos, "s_neg": s_neg}

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
        alpha_iso, eta = self.alpha_eta(z, f_sub, f_val, delta["alpha"],
                                        delta["eta"])
        sqrt_alpha = (g_el * torch.sqrt(alpha_iso + EPS_ALPHA))[:, None, None]
        if self.alpha_anisotropy:
            sqrt_alpha = sqrt_alpha * physics.expm_sym(0.5 * tens[..., -1])
        else:
            sqrt_alpha = sqrt_alpha * torch.eye(3, dtype=pd, device=z.device)

        def amp(name):
            return torch.exp(getattr(self, f"log_{name}0")[z].to(pd) + gz[name]
                             + delta[name])

        beta = amp("beta")
        raw_def = self.beta_def_head(feats).to(pd)[:, 0]
        d_def = BETA_DEF_BOUND * torch.tanh(raw_def / BETA_DEF_BOUND)
        beta_def = beta * torch.exp(self.beta_def_guess(z).to(pd)[:, 0] + d_def)
        if self.sr_shell:
            cap = torch.tensor(SR_SHELL_CAP, dtype=pd, device=raw.device)
            d_sr = g_el[:, None] * norm_cap(self.sr_vec_readout(u).to(pd), cap,
                                            (1,))[..., 0]
            Q_sr = g_el[:, None, None] * norm_cap(self.sr_ten_readout(Q).to(pd),
                                                  cap, (1, 2))[..., 0]
        else:
            d_sr, Q_sr = torch.zeros_like(mu), torch.zeros_like(th)
        if not self.atomic_dipoles:
            mu, d_sr = torch.zeros_like(mu), torch.zeros_like(d_sr)
        if not self.atomic_quadrupoles:
            th, Q_sr = torch.zeros_like(th), torch.zeros_like(Q_sr)
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

    def alpha_eta(self, z: Tensor, f_sub: Tensor, f_val: Tensor, d_alpha: Tensor,
                  d_eta: Tensor) -> tuple[Tensor, Tensor]:
        """(alpha_iso, eta) per atom from the shell fillings and the bounded
        head offsets; ``element_alpha_eta`` drops the filling dependence."""
        pd = self.physics_dtype
        if self.element_alpha_eta:
            return (self.alpha_free[z].to(pd) * torch.exp(d_alpha),
                    self.eta_free[z].to(pd) * torch.exp(d_eta))
        g_sub, g_val = self.gammas()
        alpha_iso = (self.alpha_core[z].to(pd)
                     + self.d_alpha_sub[z].to(pd) * psi(f_sub, g_sub[z].to(pd))
                     + self.d_alpha_val[z].to(pd) * psi(f_val, g_val[z].to(pd))
                     ) * torch.exp(d_alpha)
        ft = torch.tanh(f_val / 2.0) / math.tanh(0.5)
        e_cat, e_free = self.eta_cat[z].to(pd), self.eta_free[z].to(pd)
        return alpha_iso, e_cat * (e_free / e_cat) ** ft * torch.exp(d_eta)

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
            self._beta_min(si), self._beta_min(sj)), self.point_x, self.kernel_mode,
            r, rhat, si, sj)
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
        if self.transverse_invariants:
            omega = physics.transverse_invariants(rhat, vi[:, :, 2], ti[..., 2],
                                                  vj[:, :, 2], tj[..., 2])
        else:
            omega = r.new_zeros((r.shape[0], 4))
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
        if not self.charge_transfer:
            ct = torch.zeros_like(ct)
        dg = self.dispersion_globals()
        disp, c6 = physics.dispersion_pairs(
            r, at["alpha_iso"][i], at["alpha_iso"][j], at["eta"][i], at["eta"][j],
            self.r42[z[i]].to(pd), self.r42[z[j]].to(pd),
            self.alpha_free[z[i]].to(pd), self.alpha_free[z[j]].to(pd),
            dg["a1"].to(pd), dg["a2"].to(pd), dg["s8"].to(pd), s_cut=sc)
        # permanent fields: probes (i, j) <- sources (j, i), both in one call
        probe, source = torch.cat([i, j]), torch.cat([j, i])
        r2, rh2 = torch.cat([r, r]), torch.cat([-rhat, rhat])
        if self.pol_field == "net_multipole":
            c_dir = self.pol_field_scale()
            bp_f = at["beta_pol"] * c_dir
            bp_p, bp_s = bp_f[probe], bp_f[source]
            f = physics.by_range(
                physics.permanent_field_net,
                r2 * torch.minimum(bp_p, bp_s),
                self.point_x, self.kernel_mode,
                r2, rh2, bp_p, bp_s,
                at["q"][source].to(pd), at["mu"][source], at["th"][source],
            )
        else:
            src, bp = self._side(source, zf, at), at["beta_pol"][probe]
            f = physics.by_range(physics.permanent_field,
                                 r2 * torch.minimum(bp, self._beta_min(src)),
                                 self.point_x, self.kernel_mode,
                                 r2, rh2, bp, src)
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
      right;
    * ``specialize_float``: with ``dynamic=True``, torch >= 2.14 Dynamo lifts
      Python float constants (such as ``X_SWITCH``, ``H2KCAL``, and ``psi``'s
      ``eps``) to symbolic-float / 0-d tensor inputs whose ``.item()`` Inductor
      fails to lower (``KeyError: 'unbacked_bindings'``).  These floats never
      change, so baking them in as constants is exact.

    One graph then serves every chunk shape.  Parameters stay ``model``'s."""

    def __init__(self, model: FF2Model, **options):
        super().__init__()
        self.model = model
        self._compiled = torch.compile(model, **{"fullgraph": True, "dynamic": True,
                                                 **options})

    def forward(self, batch: dict):
        from torch.fx.experimental import _config as fx_config
        if self.model.so3lr_backbone:
            # FF-3 C/D: ``edges`` as a (2, E) view of an (E, 2) buffer, strides
            # (1, 2) for every E.  With the contiguous (E, 1) layout the
            # inductor backward of these arms (torch 2.10) baked the first
            # chunk's stride and failed on the next edge count.
            batch = {**batch, "edges": batch["edges"].t().contiguous().t()}
        with fx_config.patch(use_duck_shape=False, backed_size_oblivious=True), \
                torch._dynamo.config.patch(specialize_float=True), \
                torch.backends.opt_einsum.flags(enabled=False):
            return self._compiled(batch)


def compile_ff2(model: FF2Model, **options) -> CompiledFF2:
    """The compiled FF-2 forward (see :class:`CompiledFF2`)."""
    return CompiledFF2(model, **options)
