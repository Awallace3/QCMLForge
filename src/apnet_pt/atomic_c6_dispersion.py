"""Learned atomic-C6 intermolecular dispersion, without D3.

Each atom carries a C6 coefficient predicted from its environment,

    C6_i   = exp(c_Z + d6_i)                      c_Z per element, d6_i from an MPNN
    C6_ij  = sqrt(C6_i C6_j)
    C8_ij  = 3 C6_ij sqrt(Q_i Q_j) exp((d8_i + d8_j) / 2)
    C10_ij = 49/40 C8_ij^2 / C6_ij exp((d10_i + d10_j) / 2)
    E      = -sum_{i in A, j in B} sum_n s_n f_n(r_ij) C_n,ij / r_ij^n     (s6 = 1)

with damping f_n either

    "bj": r^n / (r^n + R0^n),  R0 = a1 sqrt(C8/C6) + a2   (bohr), or
    "tt": Tang-Toennies 1 - exp(-x) sum_{k<=n} x^k / k!,  x = beta_ij r, with
          beta "r0":   beta_ij = b / sqrt(C8/C6)
          beta "atom": beta_ij = sqrt(beta_i beta_j),
                       beta_i = b / (sqrt(3) sqrt(Q_i)) exp(dbeta_i)

and optionally the Slater-ISA argument x -> x - (2x^2 + 3x) / (x^2 + 3x + 3).

c_Z starts at the free-atom C6 of ``apnet_pt/data/vdw-params.csv`` (column
"C6(TS)").  The only D3 quantity used is the element constant sqrt(Q_i) =
<r4>/<r2> (``qcml_dftd3.data.r4r2``), which sets the C8/C6 ratio.  The MPNN
readouts start at exactly zero, so an untrained model is its element model;
``prefit`` fits that element model (c_Z and the damping globals) before the
MPNN trains.  The MPNN's per-element guess layer is frozen at zero, so the
deltas depend only on the environment and element C6 lives only in c_Z.

Options
-------
higher_order
    "tied" keeps C8/C10 fixed multiples of C6; "delta" adds bounded per-atom
    d8/d10 from the MPNN; "element" adds one bounded learned log-factor per
    element, so C8/C10 can move with conformation only through C6.
c6_anchor
    "free-atom-ratio" freezes c_Z at the free-atom log C6 and learns one
    global log-scale instead, so elements cannot trade C6 with each other.

d6 is unbounded; d8, d10 and dbeta are bounded by log(2) tanh(raw / log 2).
Energies are in kcal/mol, coordinates in angstrom, coefficients in atomic
units.
"""

from __future__ import annotations

import csv
import math
from importlib import resources
from types import SimpleNamespace

import torch
from torch import nn

from .AtomModels.ap3_atomtype_mpnn import AtomTypeParamMPNN

DTYPE = torch.float64
BOHR = 0.52917721067  # angstrom
H2KCAL = 627.509474
DELTA_BOUND = math.log(2.0)
ZMAX = 87
DAMPINGS = ("bj", "tt")
BETAS = ("r0", "atom")
HIGHER_ORDERS = ("tied", "delta", "element")
ANCHORS = (None, "free-atom-ratio")
LOSSES = ("huber", "huber+relative")


def free_atom_c6() -> torch.Tensor:
    """Free-atom C6 (Eh a0^6) indexed by Z, NaN where the table has none."""
    table = torch.full((ZMAX,), float("nan"), dtype=DTYPE)
    path = resources.files("apnet_pt").joinpath("data", "vdw-params.csv")
    with path.open() as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            if z < ZMAX and row["C6(TS)"]:
                table[z] = float(row["C6(TS)"])
    return table


def sqrt_q() -> torch.Tensor:
    """D3's sqrt(Q_Z) = <r4>/<r2> by Z (the C8/C6 ratio constant)."""
    from qcml_dftd3.data import r4r2

    return r4r2.R4R2(dtype=DTYPE)[:ZMAX].clone()


def tang_toennies(n: int, x: torch.Tensor) -> torch.Tensor:
    """Tang-Toennies damping f_n(x) = 1 - exp(-x) sum_{k<=n} x^k / k!."""
    term = torch.ones_like(x)
    total = torch.ones_like(x)
    for k in range(1, n + 1):
        term = term * x / k
        total = total + term
    return 1.0 - torch.exp(-x) * total


def bounded(raw: torch.Tensor) -> torch.Tensor:
    """log(2) tanh(raw / log 2): a log-factor within [1/2, 2], slope 1 at 0."""
    return DELTA_BOUND * torch.tanh(raw / DELTA_BOUND)


def _zero_offset_mpnn(n_params: int, options: dict) -> AtomTypeParamMPNN:
    """An AtomTypeParamMPNN whose output is exactly zero until it trains."""
    model = AtomTypeParamMPNN(
        n_params=n_params,
        param_start_mean=[0.0] * n_params,
        param_start_std=[0.0] * n_params,
        **options,
    ).to(DTYPE)
    with torch.no_grad():
        for stages in model.param_readout_layers:
            for stage in stages:
                final = [m for m in stage if isinstance(m, nn.Linear)][-1]
                final.weight.zero_()
                final.bias.zero_()
    model.guess_layer.requires_grad_(False)
    return model


class AtomicC6Dispersion(nn.Module):
    """Intermolecular dispersion from learned per-atom C6 coefficients.

    Parameters
    ----------
    damping : {"bj", "tt"}
        Becke-Johnson or Tang-Toennies damping.
    orders : tuple of int
        (6, 8) or (6, 8, 10).
    higher_order : {"tied", "delta", "element"}
        How C8/C10 follow C6 (see the module docstring).
    beta : {"r0", "atom"} or None
        Tang-Toennies range parameter; None means "atom" for "tt" and is
        required for "bj".
    mpnn : bool
        Predict per-atom deltas with an MPNN; False gives the element model.
    c6_anchor : {None, "free-atom-ratio"}
        Freeze c_Z at the free-atom C6 and learn one global scale.
    slater_isa : bool
        Use the Slater-ISA Tang-Toennies argument.
    mpnn_options : dict, optional
        Keyword arguments for :class:`AtomTypeParamMPNN` (n_message, n_rbf,
        n_neuron, n_embed, r_cut).
    """

    def __init__(
        self,
        damping: str = "tt",
        orders: tuple[int, ...] = (6, 8, 10),
        higher_order: str = "tied",
        beta: str | None = None,
        mpnn: bool = True,
        c6_anchor: str | None = None,
        slater_isa: bool = False,
        mpnn_options: dict | None = None,
    ):
        super().__init__()
        orders = tuple(orders)
        if damping == "tt" and beta is None:
            beta = "atom"
        self._check(damping, orders, higher_order, beta, mpnn, c6_anchor, slater_isa)
        self.damping, self.orders, self.higher_order = damping, orders, higher_order
        self.beta, self.use_mpnn, self.c6_anchor = beta, mpnn, c6_anchor
        self.slater_isa = slater_isa
        self.mpnn_options = dict(mpnn_options or {})
        self.outputs = self._mpnn_outputs()
        self.mpnn = (
            _zero_offset_mpnn(len(self.outputs), self.mpnn_options)
            if self.outputs
            else None
        )
        free = free_atom_c6()
        self.register_buffer("free_c6", free)
        self.register_buffer("sqrt_q", sqrt_q())
        self.log_c6 = nn.Parameter(
            torch.nan_to_num(free.log(), nan=0.0), requires_grad=c6_anchor is None
        )
        self.element_hi = None
        if higher_order == "element":
            names = ["d8"] + (["d10"] if 10 in orders else [])
            self.element_hi = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(ZMAX, dtype=DTYPE)) for k in names}
            )
        self.log_globals = nn.ParameterDict(
            {
                k: nn.Parameter(torch.tensor(math.log(v), dtype=DTYPE))
                for k, v in self._global_inits().items()
            }
        )

    @staticmethod
    def _check(damping, orders, higher_order, beta, mpnn, c6_anchor, slater_isa):
        if damping not in DAMPINGS:
            raise ValueError(f"damping must be one of {DAMPINGS}, got {damping!r}")
        if orders not in ((6, 8), (6, 8, 10)):
            raise ValueError(f"orders must be (6, 8) or (6, 8, 10), got {orders}")
        if higher_order not in HIGHER_ORDERS:
            raise ValueError(f"higher_order must be one of {HIGHER_ORDERS}")
        if damping == "tt" and beta not in BETAS:
            raise ValueError(f"tt damping needs beta in {BETAS}, got {beta!r}")
        if damping == "bj" and (beta is not None or slater_isa):
            raise ValueError("beta and slater_isa are Tang-Toennies options")
        if c6_anchor not in ANCHORS:
            raise ValueError(f"c6_anchor must be one of {ANCHORS}")
        if not mpnn and (higher_order == "delta" or beta == "atom"):
            raise ValueError("per-atom higher orders or beta need the MPNN")

    def _mpnn_outputs(self) -> list[str]:
        if not self.use_mpnn:
            return []
        out = ["d6"]
        if self.higher_order == "delta":
            out += ["d8"] + (["d10"] if 10 in self.orders else [])
        if self.beta == "atom":
            out.append("dbeta")
        return out

    def _global_inits(self) -> dict[str, float]:
        init = {"s8": 1.0}
        if self.c6_anchor:
            init["c6_scale"] = 1.0
        if 10 in self.orders:
            init["s10"] = 0.5
        if self.damping == "bj":
            init.update(a1=0.4, a2=2.5)
        else:
            init["b"] = 7.0
        return init

    def config(self) -> dict:
        """Constructor keyword arguments, for checkpoint round trips."""
        return {
            "damping": self.damping,
            "orders": list(self.orders),
            "higher_order": self.higher_order,
            "beta": self.beta,
            "mpnn": self.use_mpnn,
            "c6_anchor": self.c6_anchor,
            "slater_isa": self.slater_isa,
            "mpnn_options": dict(self.mpnn_options),
        }

    # ------------------------------------------------------------ parameters
    def globals(self) -> dict[str, torch.Tensor]:
        """Damping and scale globals (s8, s10, a1/a2 or b, c6_scale)."""
        return {k: v.exp() for k, v in self.log_globals.items()}

    def element_log_c6(self) -> torch.Tensor:
        """Element log C6 by Z (c_Z, plus the global scale when anchored)."""
        if self.c6_anchor:
            return self.log_c6 + self.log_globals["c6_scale"]
        return self.log_c6

    def element_parameters(self) -> list[nn.Parameter]:
        """Trainable element-level parameters: c_Z, globals, element factors."""
        hi = list(self.element_hi.values()) if self.element_hi is not None else []
        params = (self.log_c6, *self.log_globals.values(), *hi)
        return [p for p in params if p.requires_grad]

    def prefit_parameters(self) -> list[nn.Parameter]:
        """Element parameters ``prefit`` moves (element C8/C10 factors stay 0)."""
        hi = self.element_hi.values() if self.element_hi is not None else ()
        skip = {id(p) for p in hi}
        return [p for p in self.element_parameters() if id(p) not in skip]

    # --------------------------------------------------------------- forward
    def atom_terms(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        edge_index: torch.Tensor,
        molecule_ind: torch.Tensor,
        use_mpnn: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Per-atom log C6 and the bounded deltas d6, d8, d10, dbeta.

        ``edge_index`` holds intramolecular edges within ``r_cut`` and
        ``molecule_ind`` the monomer index of each atom (the MPNN's graph).
        """
        n = len(z)
        raw = {}
        if self.mpnn is not None and use_mpnn:
            graph = SimpleNamespace(
                x=z, R=R, edge_index=edge_index, molecule_ind=molecule_ind
            )
            out = self.mpnn(graph).to(DTYPE).reshape(n, len(self.outputs))
            raw = {name: out[:, k] for k, name in enumerate(self.outputs)}
        zero = torch.zeros(n, dtype=DTYPE, device=z.device)
        d6 = raw.get("d6", zero)
        if self.element_hi is not None:
            raw.update({k: v[z] for k, v in self.element_hi.items()})
        return {
            "log_c6": self.element_log_c6()[z] + d6,
            "d6": d6,
            "d8": bounded(raw["d8"]) if "d8" in raw else zero,
            "d10": bounded(raw["d10"]) if "d10" in raw else zero,
            "dbeta": bounded(raw["dbeta"]) if "dbeta" in raw else zero,
        }

    def pair_energies(
        self,
        terms: dict[str, torch.Tensor],
        z: torch.Tensor,
        R: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        """Damped dispersion energy (kcal/mol, <= 0) of each (i, j) pair."""
        p = self.globals()
        Rb = R.to(DTYPE) / BOHR
        r = (Rb[pair_j] - Rb[pair_i]).norm(dim=-1)
        log_c6 = terms["log_c6"]
        c6 = torch.exp(0.5 * (log_c6[pair_i] + log_c6[pair_j]))
        sq = self.sqrt_q[z]
        d8 = terms["d8"]
        qq = 3.0 * sq[pair_i] * sq[pair_j] * torch.exp(0.5 * (d8[pair_i] + d8[pair_j]))
        coef = {6: c6, 8: c6 * qq}
        if 10 in self.orders:
            d10 = terms["d10"]
            coef[10] = (49.0 / 40.0) * c6 * qq * qq * torch.exp(
                0.5 * (d10[pair_i] + d10[pair_j])
            )
        scale = {6: 1.0, 8: p["s8"], 10: p.get("s10")}
        if self.damping == "bj":
            r0 = p["a1"] * qq.sqrt() + p["a2"]
            pair = sum(scale[n] * coef[n] / (r**n + r0**n) for n in self.orders)
        else:
            x = self._tt_beta(terms, p, sq, qq, pair_i, pair_j) * r
            if self.slater_isa:
                x = x - (2 * x * x + 3 * x) / (x * x + 3 * x + 3)
            pair = sum(
                scale[n] * tang_toennies(n, x) * coef[n] / r**n for n in self.orders
            )
        return -H2KCAL * pair

    def _tt_beta(self, terms, p, sq, qq, pair_i, pair_j):
        if self.beta == "r0":
            return p["b"] / qq.sqrt()
        beta_i = p["b"] / (math.sqrt(3.0) * sq) * torch.exp(terms["dbeta"])
        return (beta_i[pair_i] * beta_i[pair_j]).sqrt()

    def forward(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        edge_index: torch.Tensor,
        molecule_ind: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        pair_dimer: torch.Tensor,
        n_dimer: int,
        use_mpnn: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Dispersion energy per dimer.

        Parameters
        ----------
        z, R : Tensor
            Atomic numbers and coordinates (angstrom) of every atom in the batch.
        edge_index, molecule_ind : Tensor
            Intramolecular graph for the MPNN.
        pair_i, pair_j : Tensor
            Atom indices of every intermolecular pair (i in A, j in B).
        pair_dimer : Tensor
            Dimer index of each pair.
        n_dimer : int
            Number of dimers in the batch.

        Returns
        -------
        energy : Tensor
            (n_dimer,) dispersion energy in kcal/mol.
        extras : dict
            ``"atom"`` (per-atom terms from :meth:`atom_terms`) and ``"pair"``
            (per-pair energies).
        """
        terms = self.atom_terms(z, R, edge_index, molecule_ind, use_mpnn)
        pair = self.pair_energies(terms, z, R, pair_i, pair_j)
        energy = pair.new_zeros(n_dimer).index_add(0, pair_dimer, pair)
        return energy, {"atom": terms, "pair": pair}


def dispersion_objective(
    pred: torch.Tensor, target: torch.Tensor, loss: dict
) -> torch.Tensor:
    """Huber(delta), plus for "huber+relative" relative_weight x
    Huber((p - t) / (|t| + relative_floor), relative_delta).

    The relative term gives long-range dimers (|E| ~ 0.05-0.5 kcal/mol, about
    1% of an absolute Huber loss) a share of the gradient.
    """
    if loss["name"] not in LOSSES:
        raise ValueError(f"loss name must be one of {LOSSES}, got {loss['name']!r}")
    value = nn.functional.huber_loss(pred, target, delta=loss["delta"])
    if loss["name"] == "huber+relative":
        rel = (pred - target) / (target.abs() + loss["relative_floor"])
        value = value + loss["relative_weight"] * nn.functional.huber_loss(
            rel, torch.zeros_like(rel), delta=loss["relative_delta"]
        )
    return value


def prefit(
    model: AtomicC6Dispersion,
    inputs: dict,
    target: torch.Tensor,
    loss: dict,
    steps: int = 1000,
    lr: float = 0.03,
) -> dict:
    """Fit the element model (c_Z and damping globals) with the MPNN off.

    Full-batch Adam on ``model(**inputs, use_mpnn=False)`` from the model's
    current values; lr drops 5x at 60% and 85% of ``steps``.  The best values
    are loaded into the model.  Deterministic for fixed inputs, so every model
    of one damping form starts from the same element model.

    Returns the first and best loss, the step count and the L1 distance the
    parameters moved.
    """
    if steps < 1:
        raise ValueError(f"prefit needs steps >= 1, got {steps}")
    params = model.prefit_parameters()
    start = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=lr)
    best_loss, best = math.inf, None
    first = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred, _ = model(**inputs, use_mpnn=False)
        value = dispersion_objective(pred, target, loss)
        if not torch.isfinite(value):
            raise FloatingPointError(f"prefit loss is non-finite at step {step}")
        v = float(value.detach())
        first = v if first is None else first
        if v < best_loss:  # the values that produced v, before this step
            best_loss, best = v, [p.detach().clone() for p in params]
        value.backward()
        opt.step()
        if step in (int(0.6 * steps), int(0.85 * steps)):
            for group in opt.param_groups:
                group["lr"] *= 0.2
    with torch.no_grad():
        for p, v in zip(params, best):
            p.copy_(v)
    moved = float(sum((p.detach() - s).abs().sum() for p, s in zip(params, start)))
    return {
        "prefit_loss_first": first,
        "prefit_loss_best": best_loss,
        "steps": steps,
        "param_l1_moved": moved,
    }
