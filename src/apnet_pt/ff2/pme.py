"""Cutoff, neighbor-list, and SPME long-range engine for FF-2 MD.

Provides :class:`FF2LongRangeEnergy` and :func:`compile_long_range_energy`,
replacing FF-2's O(N^2) all-pairs lists and O(N^3) dense Cholesky polarization
with:

* O(N) cell-linked neighbor lists with static capacity and Verlet skin
  (:class:`torch_pes.NeighborList`);
* Smooth Particle Mesh Ewald up to quadrupoles (:class:`torch_pes.MultipolePME`)
  for periodic electrostatics and permanent electric fields (in one FFT), plus
  exact intramonomer ``erf`` exclusions;
* Matrix-free preconditioned CG polarization using sparse real-space pair
  matvecs plus reusable SPME dipole-dipole matvecs (envelope theorem forces);
* Dispersion long range supporting low-rank Chebyshev--Nyström Casimir--Polder
  LJPME, geometric-mean LJPME, or homogeneous tail correction
  (:class:`torch_pes.DispersionPME`);
* OpenMM 8.5 ``PythonForce`` helper (:func:`openmm_python_force`) passing both
  positions and periodic box vectors per step.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from torch_pes.calculators.dispersion_pme import (
    DispersionPME,
    homogeneous_tail_correction,
)
from torch_pes.calculators.multipole_pme import MultipolePME, SPMEState
from torch_pes.calculators.neighbor_list import NeighborBuffers, NeighborList
from torch_pes.lib.boys import (
    dipole_tensor_coeffs,
    ewald_boys_erf,
    multipole_pair_energy,
    multipole_pair_field,
)
from torch_pes.tuning.multipole import tune_dispersion_pme, tune_multipole_pme

from ..ff1.model import H2KCAL
from ..ff1.trunk import ANG2BOHR
from . import kernels as K
from . import physics
from .model import FF2Model

NM2ANG = 10.0
KCAL2KJ = 4.184
NM2BOHR = NM2ANG * ANG2BOHR


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1)


def _mat_vec(T: Tensor, v: Tensor) -> Tensor:
    return (T * v[..., None, :]).sum(dim=-1)


def _quintic_switch(r: Tensor, r_on: float, r_cut: float) -> Tensor:
    """C^2 quintic switching function: 1 for ``r <= r_on``, 0 for ``r >= r_cut``."""
    u = ((r - r_on) / (r_cut - r_on)).clamp(0.0, 1.0)
    return 1.0 - u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _cell_inverse(cell: Tensor) -> Tensor:
    if cell.is_cuda:
        return torch.linalg.inv_ex(cell)[0]
    return torch.linalg.inv(cell)


def _build_topology_buffers(
    z: Sequence[int],
    monomer: Sequence[int],
    monomer_charge: Sequence[float],
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Build static O(N) intramonomer topology buffers (never O(N^2)!).

    Constructs on CPU to avoid per-monomer device syncs on GPU, then transfers
    to ``device``.
    """
    z_t = torch.as_tensor(z, dtype=torch.long, device="cpu")
    mon = torch.as_tensor(monomer, dtype=torch.long, device="cpu")
    q_mon = torch.as_tensor(monomer_charge, dtype=torch.float64, device="cpu")
    n = int(z_t.shape[0])
    n_mon = int(mon.max().item()) + 1 if n > 0 else 0
    if mon.shape != (n,) or q_mon.shape[0] != n_mon:
        raise ValueError(
            "monomer must label N atoms with 0 .. M-1, and "
            "monomer_charge must have M entries"
        )

    # Build intramonomer directed edges (src != dst) and half-list exclusions (i < j)
    # in O(sum_m n_m^2) = O(N), NEVER allocating an (N, N) matrix!
    edge_src_parts: list[Tensor] = []
    edge_dst_parts: list[Tensor] = []
    excl_i_parts: list[Tensor] = []
    excl_j_parts: list[Tensor] = []
    anchor = torch.zeros(n, dtype=torch.long, device="cpu")

    order = torch.argsort(mon, stable=True)
    sorted_mon = mon[order]
    counts = torch.bincount(sorted_mon, minlength=n_mon)
    starts = torch.cumsum(counts, dim=0) - counts

    for m_idx in range(n_mon):
        c = int(counts[m_idx].item())
        if c == 0:
            continue
        s = int(starts[m_idx].item())
        atoms = order[s : s + c]
        anchor[atoms] = atoms[0]
        if c > 1:
            ii, jj = torch.triu_indices(c, c, offset=1, device="cpu")
            ai_m, aj_m = atoms[ii], atoms[jj]
            excl_i_parts.append(ai_m)
            excl_j_parts.append(aj_m)
            edge_src_parts.append(torch.cat([ai_m, aj_m]))
            edge_dst_parts.append(torch.cat([aj_m, ai_m]))

    if excl_i_parts:
        excl_i = torch.cat(excl_i_parts)
        excl_j = torch.cat(excl_j_parts)
        edges = torch.stack([torch.cat(edge_src_parts), torch.cat(edge_dst_parts)])
    else:
        empty = torch.zeros(0, dtype=torch.long, device="cpu")
        excl_i, excl_j = empty, empty
        edges = torch.zeros((2, 0), dtype=torch.long, device="cpu")

    return {
        "z": z_t.to(device),
        "edges": edges.to(device),
        "monomer": mon.to(device),
        "monomer_charge": q_mon.to(device),
        "monomer_counts": counts.to(dtype=torch.float64, device=device),
        "excl_i": excl_i.to(device),
        "excl_j": excl_j.to(device),
        "anchor": anchor.to(device),
    }


def _unwrap_monomers(R_ang: Tensor, anchor: Tensor, cell_ang: Tensor) -> Tensor:
    """Shift each atom by an integer lattice vector into the minimum-image
    neighborhood of its monomer's anchor atom (exact identity derivative)."""
    inv_cell = _cell_inverse(cell_ang)
    frac_diff = (R_ang - R_ang[anchor]) @ inv_cell
    shift = torch.round(frac_diff).detach()
    return R_ang - shift @ cell_ang


def _sparse_dipole_matvec(
    p: Tensor,
    pk: Tensor,
    pl: Tensor,
    rhat: Tensor,
    l33: Tensor,
    l53: Tensor,
) -> Tensor:
    """Apply symmetric sparse real-space dipole tensor
    :math:`T^{\\mathrm{pol}}_{kl} = l_{33} I - 3 l_{53} \\hat{r}\\hat{r}^T`
    to ``p`` of shape ``(N, 3)``."""
    pk_vec = p[pk]
    pl_vec = p[pl]
    # On atom k from l: l33 * p_l - 3 l53 (p_l . rhat) rhat
    tk = l33[:, None] * pl_vec - (3.0 * l53 * _dot(pl_vec, rhat))[:, None] * rhat
    tl = l33[:, None] * pk_vec - (3.0 * l53 * _dot(pk_vec, rhat))[:, None] * rhat
    out = p.new_zeros(p.shape)
    out = out.index_add(0, pk, tk)
    out = out.index_add(0, pl, tl)
    return out


def _sparse_cg_solve(
    sqrt_alpha: Tensor,
    rhs: Tensor,
    pk: Tensor,
    pl: Tensor,
    rhat: Tensor,
    l33_sr: Tensor,
    l53_sr: Tensor,
    n_iter: int,
    pme: MultipolePME | None = None,
    spme_state: SPMEState | None = None,
) -> Tensor:
    """Fixed-step CG solve of ``(I + S T_pol S) w = rhs`` with ZERO dense
    ``(3N, 3N)`` matrices.

    Runs inside ``torch.no_grad()``; the diagonal 3x3 blocks of ``I + K`` are
    identically ``I_3``, so unpreconditioned CG is already block-Jacobi scaled.
    """
    S = sqrt_alpha.detach()
    r_h = rhat.detach()
    c33 = l33_sr.detach()
    c53 = l53_sr.detach()
    b = rhs.detach()

    # Detached SPME state so no autograd graph is built inside CG
    st_det: SPMEState | None = None
    if pme is not None and spme_state is not None:
        st_det = SPMEState(
            ns_mesh=spme_state.ns_mesh,
            order=spme_state.order,
            volume=spme_state.volume.detach(),
            jacobian=spme_state.jacobian.detach(),
            flat_idx=spme_state.flat_idx,
            w0=spme_state.w0.detach(),
            w1=spme_state.w1.detach(),
            w2=None,
            k_kernel=spme_state.k_kernel.detach(),
        )

    def matvec(v: Tensor) -> Tensor:
        p_vec = _mat_vec(S, v)
        tp = _sparse_dipole_matvec(p_vec, pk, pl, r_h, c33, c53)
        if pme is not None and st_det is not None:
            tp = tp + pme.reciprocal_dipole_matvec(p_vec, st_det)
        return v + _mat_vec(S, tp)

    w = b.clone()
    r = b - matvec(w)
    d = r.clone()
    rr = (r * r).sum()
    tiny = torch.finfo(b.dtype).tiny
    for _ in range(n_iter):
        ad = matvec(d)
        denom = (d * ad).sum().clamp_min(tiny)
        step = rr / denom
        w = w + step * d
        r = r - step * ad
        rr_new = (r * r).sum()
        d = r + (rr_new / rr.clamp_min(tiny)) * d
        rr = rr_new
    return w


class _FF2LongRangeCore(nn.Module):
    """Pure-tensor static-shape core of :class:`FF2LongRangeEnergy`, suitable
    for ``torch.compile(fullgraph=True, dynamic=False)``."""

    def __init__(
        self,
        model: FF2Model,
        topo: dict[str, Tensor],
        r_cut_bohr: float,
        pbc: bool,
        pol_cg_iters: int,
        elst_pme: MultipolePME | None,
        disp_pme: DispersionPME | None,
        disp_mode: str,
        c8_tail: bool,
        cutoff_switch: str,
        switch_width_bohr: float,
        pair_chunk_size: int | None = None,
        state_dtype: torch.dtype = torch.float64,
        pair_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.model = model
        self.r_cut_bohr = float(r_cut_bohr)
        self.pbc = bool(pbc)
        self.pol_cg_iters = int(pol_cg_iters)
        self.elst_pme = elst_pme
        self.disp_pme = disp_pme
        self.disp_mode = disp_mode
        self.c8_tail = bool(c8_tail)
        self.cutoff_switch = cutoff_switch
        self.switch_width_bohr = float(switch_width_bohr)
        self.pair_chunk_size = (
            None if pair_chunk_size is None else int(pair_chunk_size)
        )
        self.state_dtype = state_dtype
        self.pair_dtype = pair_dtype
        for k, v in topo.items():
            self.register_buffer(f"topo_{k}", v)

    def _pair_vectors(
        self,
        R_b: Tensor,
        cell_b: Tensor | None,
        ai: Tensor,
        aj: Tensor,
        shift: Tensor,
        mask: Tensor,
        apply_rcut: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(r_safe, rhat_safe, active)`` for padded neighbor pairs."""
        disp = R_b[aj] - R_b[ai]
        if self.pbc:
            assert cell_b is not None
            disp = disp + shift.to(R_b.dtype) @ cell_b
        r2 = (disp * disp).sum(dim=-1)
        active = (
            mask & (r2 < (self.r_cut_bohr * self.r_cut_bohr))
            if apply_rcut
            else mask
        )
        # Safe nonzero distance & unit vector on inactive/padded slots so
        # kernels & autograd never see r = 0
        r_safe = torch.sqrt(torch.where(active, r2, torch.full_like(r2, 100.0)))
        unit_x = torch.zeros_like(disp)
        unit_x[..., 0] = 1.0
        rhat_safe = torch.where(active[:, None], disp / r_safe[:, None], unit_x)
        return r_safe, rhat_safe, active

    def _group_weights(
        self, R_b: Tensor, ai: Tensor, aj: Tensor, mask: Tensor, pd: torch.dtype
    ) -> Tensor:
        """Charge-neutral monomer-COM quintic switching weight per pair."""
        mon = self.topo_monomer
        counts = self.topo_monomer_counts.to(pd)
        coms = R_b.new_zeros((counts.shape[0], 3)).index_add(0, mon, R_b) / counts[
            :, None
        ]
        d_com = coms[mon[aj]] - coms[mon[ai]]
        r_com2 = (d_com * d_com).sum(dim=-1)
        # Same-monomer pairs have r_com2 == 0 -> safe sqrt with active mask
        active_g = mask & (r_com2 < (self.r_cut_bohr * self.r_cut_bohr))
        r_com = torch.sqrt(r_com2.clamp_min(1e-30))
        r_on = max(self.r_cut_bohr - self.switch_width_bohr, 0.0)
        return active_g.to(pd) * _quintic_switch(r_com, r_on, self.r_cut_bohr)

    def _c8_tail_energy(
        self,
        z: Tensor,
        at: dict[str, Tensor],
        volume: Tensor,
        pd: torch.dtype,
        dg: dict[str, Tensor],
    ) -> Tensor:
        """O(N) approximate homogeneous C8 tail correction using separable C6
        factors times per-atom ``sqrt(3) * r42 * (a / a_free)^(1/3)``."""
        eps_q = 1e-6
        af = self.model.alpha_free[z].to(pd)
        r42 = self.model.r42[z].to(pd)
        a_iso = at["alpha_iso"].to(pd)
        eta_v = at["eta"].to(pd)
        scale = math.sqrt(3.0) * r42 * (
            (a_iso + eps_q) / (af + eps_q)
        ) ** (1.0 / 3.0)
        if self.disp_pme is not None:
            fac = self.disp_pme.factors(a_iso, eta_v)
        else:
            fac = (a_iso * torch.sqrt(eta_v))[:, None]
        # sum_{i,j} C8_ij ~ || sum_i (scale_i * fac_i) ||^2
        c8_vec = (scale[:, None] * fac).sum(dim=0)
        c8_sum = (c8_vec * c8_vec).sum()
        return homogeneous_tail_correction(
            volume=volume,
            r_cut=self.r_cut_bohr,
            c6_sum=torch.zeros((), dtype=pd, device=volume.device),
            c8_sum=c8_sum,
            s8=dg["s8"].to(pd),
        )

    def _eval_inter_chunk(
        self,
        R_b: Tensor,
        cell_b: Tensor | None,
        zf: Tensor,
        z: Tensor,
        q_tot: Tensor,
        mu_tot: Tensor,
        th_tot: Tensor,
        n_act: Tensor,
        beta: Tensor,
        shell_M: Tensor,
        shell_T: Tensor,
        shell_b: Tensor,
        beta_pol: Tensor,
        vecs: Tensor,
        tens: Tensor,
        A_at: Tensor,
        B_at: Tensor,
        D_at: Tensor,
        C_at: Tensor,
        eta_B: Tensor,
        alpha_iso: Tensor,
        eta: Tensor,
        disp_factors: Tensor | None,
        a1: Tensor,
        a2: Tensor,
        s8: Tensor,
        ai: Tensor,
        aj: Tensor,
        shift: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluate all real-space intermolecular pair terms on one chunk
        ``(ai, aj, shift, mask)``, returning ``(elst_sr, exch, ct, disp_sr, field_sr)``.

        In mixed precision (``state_dtype=float64``, ``pair_dtype=float32``),
        pair geometry ``(r, rhat)``, Ewald ``erf`` subtractions, and all
        reductions stay in ``float64`` while the heavy short-range pair
        kernels (``table2d``, exchange, CT, BJ dispersion) execute in
        ``float32``.
        """
        model = self.model
        sd = self.state_dtype
        kd = self.pair_dtype
        is_group = (not self.pbc) and (self.cutoff_switch == "group")
        r, rhat, active = self._pair_vectors(
            R_b, cell_b, ai, aj, shift, mask, apply_rcut=not is_group
        )
        w_active = active.to(sd)
        if is_group:
            w_pair = self._group_weights(R_b, ai, aj, mask, sd)
        elif not self.pbc and self.cutoff_switch == "smooth":
            r_on = max(self.r_cut_bohr - self.switch_width_bohr, 0.0)
            w_pair = w_active * _quintic_switch(r, r_on, self.r_cut_bohr)
        else:
            w_pair = w_active

        # Heavy short-range pair kernels execute in pair_dtype (kd)
        r_k = r.to(kd)
        rhat_k = rhat.to(kd)
        zf_k = zf.to(kd)
        at_view = {
            "n_act": n_act.to(kd),
            "beta": beta.to(kd),
            "q": q_tot.to(kd),
            "shell_M": shell_M.to(kd),
            "shell_T": shell_T.to(kd),
            "shell_b": shell_b.to(kd),
        }
        si = model._side(ai, zf_k, at_view)
        sj = model._side(aj, zf_k, at_view)
        sc = model._s_cut(r_k)

        elst_slater = physics.by_range(
            physics.elst_pairs,
            r_k * torch.minimum(model._beta_min(si), model._beta_min(sj)),
            model.point_x,
            model.kernel_mode,
            r_k,
            rhat_k,
            si,
            sj,
        ).to(sd)

        probe = torch.cat([ai, aj])
        source = torch.cat([aj, ai])
        src_both = model._side(source, zf_k, at_view)
        r2_both = torch.cat([r, r])
        rhat_both = torch.cat([-rhat, rhat])
        r2_both_k = torch.cat([r_k, r_k])
        rhat_both_k = torch.cat([-rhat_k, rhat_k])
        bp_probe_k = beta_pol.to(kd)[probe]
        f_slater = physics.by_range(
            physics.permanent_field,
            r2_both_k * torch.minimum(bp_probe_k, model._beta_min(src_both)),
            model.point_x,
            model.kernel_mode,
            r2_both_k,
            rhat_both_k,
            bp_probe_k,
            src_both,
        )
        if sc is not None:
            f_slater = f_slater * torch.cat([sc, sc])[:, None]
        f_slater = f_slater.to(sd)

        if self.pbc:
            assert self.elst_pme is not None
            boys_erf = ewald_boys_erf(r, self.elst_pme.alpha)
            elst_erf = multipole_pair_energy(
                r,
                rhat,
                q_tot[ai],
                mu_tot[ai],
                th_tot[ai],
                q_tot[aj],
                mu_tot[aj],
                th_tot[aj],
                boys_erf,
            )
            e_elst_sr = ((elst_slater - elst_erf) * w_active).sum()
            boys_erf_both = tuple(torch.cat([b, b]) for b in boys_erf)
            f_erf = multipole_pair_field(
                r2_both,
                rhat_both,
                q_tot[source],
                mu_tot[source],
                th_tot[source],
                boys_erf_both,
            )
            f_sr_pair = (f_slater - f_erf) * torch.cat([w_active, w_active])[:, None]
            field_sr = R_b.new_zeros(R_b.shape).index_add(0, probe, f_sr_pair)
        else:
            e_elst_sr = (elst_slater * w_pair).sum()
            w_both = torch.cat([w_pair, w_pair])[:, None]
            field_sr = R_b.new_zeros(R_b.shape).index_add(0, probe, f_slater * w_both)

        # Exchange & charge transfer in pair_dtype (kd), summed in state_dtype (sd)
        vecs_k, tens_k = vecs.to(kd), tens.to(kd)
        A_k, B_k, D_k, C_k, eta_B_k = (
            A_at.to(kd),
            B_at.to(kd),
            D_at.to(kd),
            C_at.to(kd),
            eta_B.to(kd),
        )
        vi, vj, ti, tj = vecs_k[ai], vecs_k[aj], tens_k[ai], tens_k[aj]
        rq = rhat_k[:, :, None, None] * rhat_k[:, None, :, None]
        quad_i = (ti[..., :5] * rq).sum((1, 2))
        quad_j = (tj[..., :5] * rq).sum((1, 2))
        lin_i = (vi * rhat_k[:, :, None]).sum(1)
        lin_j = (vj * rhat_k[:, :, None]).sum(1)
        xi_i, xi_j = lin_i + quad_i, quad_j - lin_j
        pre_i, b_i, _, d_i, c_i = xi_i.unbind(1)
        pre_j, b_j, _, d_j, c_j = xi_j.unbind(1)
        B_ij = B_k[ai] * torch.exp(b_i)
        B_ji = B_k[aj] * torch.exp(b_j)
        omega = physics.transverse_invariants(
            rhat_k, vi[:, :, 2], ti[..., 2], vj[:, :, 2], tj[..., 2]
        )
        exch_pair = physics.exchange_pairs(
            r_k,
            rhat_k,
            A_k[ai],
            A_k[aj],
            B_ij,
            B_ji,
            pre_i,
            pre_j,
            omega,
            torch.tanh(model.kappa_ex_raw).to(kd),
            sc,
        )
        w_ct = torch.exp((omega * torch.tanh(model.kappa_ct_raw).to(kd)).sum(-1))
        gamma_sum = (
            D_k[ai] * torch.exp(d_i) * C_k[aj] * torch.exp(c_j)
            + D_k[aj] * torch.exp(d_j) * C_k[ai] * torch.exp(c_i)
        ) * w_ct
        ct_pair = physics.charge_transfer_pairs(
            r_k,
            exch_pair,
            zf_k[ai],
            zf_k[aj],
            gamma_sum,
            eta_B_k[ai] * B_ij,
            eta_B_k[aj] * B_ji,
            eps_ct=1e-30 if kd == torch.float64 else 1e-15,
            s_cut=sc,
        )
        s_gate = model.ct_grad_scale.to(kd)
        ct_pair = s_gate * ct_pair + (1.0 - s_gate) * ct_pair.detach()
        e_exch = (exch_pair.to(sd) * w_pair).sum()
        e_ct = (ct_pair.to(sd) * w_pair).sum()

        # Dispersion real space in pair_dtype (kd), Ewald correction & sum in sd
        alpha_k, eta_k = alpha_iso.to(kd), eta.to(kd)
        disp_pair, _ = physics.dispersion_pairs(
            r_k,
            alpha_k[ai],
            alpha_k[aj],
            eta_k[ai],
            eta_k[aj],
            model.r42[z[ai]].to(kd),
            model.r42[z[aj]].to(kd),
            model.alpha_free[z[ai]].to(kd),
            model.alpha_free[z[aj]].to(kd),
            a1.to(kd),
            a2.to(kd),
            s8.to(kd),
            s_cut=sc,
        )
        disp_pair = disp_pair.to(sd)
        if self.pbc and self.disp_pme is not None:
            assert disp_factors is not None
            disp_corr = self.disp_pme.real_space_correction(r, disp_factors, ai, aj)
            e_disp_sr = ((disp_pair + disp_corr) * w_active).sum()
        elif self.pbc and self.disp_mode == "tail":
            e_disp_sr = (disp_pair * w_active).sum()
        else:
            e_disp_sr = (disp_pair * w_pair).sum()

        return e_elst_sr, e_exch, e_ct, e_disp_sr, field_sr

    def _eval_pol_coeff_chunk(
        self, bk: Tensor, bl: Tensor, rr: Tensor, wp: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate real-space dipole-tensor coefficients ``(l33_sr, l53_sr)``
        for one polarization pair chunk."""
        model = self.model
        sd = self.state_dtype
        kd = self.pair_dtype
        bk_k, bl_k, rr_k = bk.to(kd), bl.to(kd), rr.to(kd)
        lam_c = physics.by_range(
            lambda a, b_, r_, mode: physics._two(a, b_, r_, K.DIPOLE_T, mode),
            rr_k * torch.minimum(bk_k, bl_k),
            model.point_x,
            model.kernel_mode,
            bk_k,
            bl_k,
            rr_k,
        )
        c33, c53 = lam_c[:, 0].to(sd), lam_c[:, 1].to(sd)
        if self.pbc:
            assert self.elst_pme is not None
            e33, e53 = dipole_tensor_coeffs(
                rr, ewald_boys_erf(rr, self.elst_pme.alpha)
            )
            return (c33 - e33) * wp, (c53 - e53) * wp
        return c33 * wp, c53 * wp

    @staticmethod
    def _pad_inter_chunk(
        ai: Tensor, aj: Tensor, shift: Tensor, mask: Tensor, chunk_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pad = chunk_size - int(ai.shape[0])
        if pad <= 0:
            return ai, aj, shift, mask
        dev = ai.device
        return (
            torch.cat([ai, torch.zeros(pad, dtype=ai.dtype, device=dev)]),
            torch.cat([aj, torch.zeros(pad, dtype=aj.dtype, device=dev)]),
            torch.cat([shift, torch.zeros((pad, 3), dtype=shift.dtype, device=dev)]),
            torch.cat([mask, torch.zeros(pad, dtype=mask.dtype, device=dev)]),
        )

    @staticmethod
    def _pad_pol_chunk(
        bk: Tensor, bl: Tensor, rr: Tensor, wp: Tensor, chunk_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pad = chunk_size - int(bk.shape[0])
        if pad <= 0:
            return bk, bl, rr, wp
        return (
            torch.cat([bk, bk.new_ones(pad)]),
            torch.cat([bl, bl.new_ones(pad)]),
            torch.cat([rr, rr.new_full((pad,), 10.0)]),
            torch.cat([wp, wp.new_zeros(pad)]),
        )

    def _forward_impl(
        self,
        positions_nm: Tensor,
        cell_nm: Tensor | None,
        inter_ai: Tensor,
        inter_aj: Tensor,
        inter_shift: Tensor,
        inter_mask: Tensor,
        pol_k: Tensor,
        pol_l: Tensor,
        pol_shift: Tensor,
        pol_mask: Tensor,
        inter_chunk_fn=None,
        pol_chunk_fn=None,
    ) -> Tensor:
        """Compute ``(4,)`` FF-2 components ``[elst, exch, ind, disp]`` in kcal/mol."""
        model = self.model
        pd = self.state_dtype
        z = self.topo_z
        zf = z.to(pd)
        R_ang = positions_nm.to(pd) * NM2ANG
        cell_ang = None if cell_nm is None else cell_nm.to(pd) * NM2ANG
        R_mono_ang = (
            _unwrap_monomers(R_ang, self.topo_anchor, cell_ang)
            if (self.pbc and cell_ang is not None)
            else R_ang
        )

        # 1. Monomer trunk + one-mu charges + atomic properties
        mono_batch = {
            "z": z,
            "R": R_mono_ang,
            "edges": self.topo_edges,
            "monomer": self.topo_monomer,
            "monomer_charge": self.topo_monomer_charge,
        }
        R_mono_b, at = model._forward_atoms(mono_batch)
        R_mono_b = R_mono_b.to(pd)
        R_b = R_ang * ANG2BOHR
        cell_b = None if cell_ang is None else cell_ang * ANG2BOHR

        q_tot = at["q"].to(pd)
        mu_tot = at["mu"].to(pd)
        th_tot = at["th"].to(pd)
        dg = model.dispersion_globals()
        a1, a2, s8 = dg["a1"].to(pd), dg["a2"].to(pd), dg["s8"].to(pd)
        disp_factors = (
            self.disp_pme.factors(at["alpha_iso"].to(pd), at["eta"].to(pd))
            if (self.pbc and self.disp_pme is not None)
            else None
        )

        # 2. Real-space intermolecular pairs (optionally gradient-checkpointed
        # in static-size chunks to bound peak activation memory on large boxes)
        common_args = (
            R_b,
            cell_b,
            zf,
            z,
            q_tot,
            mu_tot,
            th_tot,
            at["n_act"].to(pd),
            at["beta"],
            at["shell_M"],
            at["shell_T"],
            at["shell_b"],
            at["beta_pol"],
            at["vecs"],
            at["tens"],
            at["A"],
            at["B"],
            at["D"],
            at["C"],
            at["eta_B"],
            at["alpha_iso"],
            at["eta"],
            disp_factors,
            a1,
            a2,
            s8,
        )
        n_inter = inter_ai.shape[0]
        fn_inter = self._eval_inter_chunk if inter_chunk_fn is None else inter_chunk_fn
        if self.pair_chunk_size is not None and (
            inter_chunk_fn is not None or n_inter > self.pair_chunk_size
        ):
            from torch.utils.checkpoint import checkpoint

            e_elst_sr = R_b.new_zeros(())
            exch = R_b.new_zeros(())
            ct = R_b.new_zeros(())
            e_disp_sr = R_b.new_zeros(())
            field = R_b.new_zeros(R_b.shape)
            step = self.pair_chunk_size
            pad_static = inter_chunk_fn is not None
            for s in range(0, n_inter, step):
                e_idx = min(s + step, n_inter)
                ai_c = inter_ai[s:e_idx]
                aj_c = inter_aj[s:e_idx]
                sh_c = inter_shift[s:e_idx]
                mk_c = inter_mask[s:e_idx]
                if pad_static:
                    ai_c, aj_c, sh_c, mk_c = self._pad_inter_chunk(
                        ai_c, aj_c, sh_c, mk_c, step
                    )
                c_out = checkpoint(
                    fn_inter,
                    *common_args,
                    ai_c,
                    aj_c,
                    sh_c,
                    mk_c,
                    use_reentrant=False,
                )
                e_elst_sr = e_elst_sr + c_out[0]
                exch = exch + c_out[1]
                ct = ct + c_out[2]
                e_disp_sr = e_disp_sr + c_out[3]
                field = field + c_out[4]
        else:
            e_elst_sr, exch, ct, e_disp_sr, field = fn_inter(
                *common_args, inter_ai, inter_aj, inter_shift, inter_mask
            )

        # 3. Reciprocal-space multipolar SPME & intramonomer exclusions
        spme_state: SPMEState | None = None
        if self.pbc:
            assert self.elst_pme is not None and cell_b is not None
            e_elst_rec, field_rec, spme_state = (
                self.elst_pme.reciprocal_energy_and_field(
                    R_b, cell_b, q_tot, mu_tot, th_tot
                )
            )

            ex_i, ex_j = self.topo_excl_i, self.topo_excl_j
            if ex_i.numel() > 0:
                r_ex, rhat_ex = K.safe_unit(R_mono_b[ex_j] - R_mono_b[ex_i])
                boys_ex = ewald_boys_erf(r_ex, self.elst_pme.alpha)
                e_elst_excl = multipole_pair_energy(
                    r_ex,
                    rhat_ex,
                    q_tot[ex_i],
                    mu_tot[ex_i],
                    th_tot[ex_i],
                    q_tot[ex_j],
                    mu_tot[ex_j],
                    th_tot[ex_j],
                    boys_ex,
                ).sum()
                f_ex_j = multipole_pair_field(
                    r_ex, rhat_ex, q_tot[ex_i], mu_tot[ex_i], th_tot[ex_i], boys_ex
                )
                f_ex_i = multipole_pair_field(
                    r_ex, -rhat_ex, q_tot[ex_j], mu_tot[ex_j], th_tot[ex_j], boys_ex
                )
                field_excl = (
                    R_b.new_zeros(R_b.shape)
                    .index_add(0, ex_j, f_ex_j)
                    .index_add(0, ex_i, f_ex_i)
                )
            else:
                e_elst_excl = R_b.new_zeros(())
                field_excl = R_b.new_zeros(R_b.shape)

            elst = e_elst_sr + e_elst_rec - e_elst_excl
            field = field + field_rec - field_excl
        else:
            elst = e_elst_sr

        # 4. Reciprocal-space dispersion PME / tail & exclusions
        if self.pbc and self.disp_pme is not None:
            assert cell_b is not None and disp_factors is not None
            e_disp_rec = self.disp_pme.reciprocal_energy(R_b, cell_b, disp_factors)
            ex_i, ex_j = self.topo_excl_i, self.topo_excl_j
            if ex_i.numel() > 0:
                r_ex, _ = K.safe_unit(R_mono_b[ex_j] - R_mono_b[ex_i])
                e_disp_excl = self.disp_pme.real_space_correction(
                    r_ex, disp_factors, ex_i, ex_j
                ).sum()
            else:
                e_disp_excl = R_b.new_zeros(())
            disp = e_disp_sr + e_disp_rec + e_disp_excl
            if self.c8_tail:
                vol = torch.abs(torch.linalg.det(cell_b))
                disp = disp + self._c8_tail_energy(z, at, vol, pd, dg)
        elif self.pbc and self.disp_mode == "tail":
            assert cell_b is not None
            vol = torch.abs(torch.linalg.det(cell_b))
            c_g = at["alpha_iso"].to(pd) * torch.sqrt(at["eta"].to(pd))
            c6_sum = c_g.sum() ** 2
            disp = e_disp_sr + homogeneous_tail_correction(
                vol, self.r_cut_bohr, c6_sum
            )
            if self.c8_tail:
                disp = disp + self._c8_tail_energy(z, at, vol, pd, dg)
        else:
            disp = e_disp_sr

        # 5. Sparse (+ SPME) mutual polarization with envelope theorem
        is_group = (not self.pbc) and (self.cutoff_switch == "group")
        r_p, rhat_p, active_p = self._pair_vectors(
            R_b,
            cell_b,
            pol_k,
            pol_l,
            pol_shift,
            pol_mask,
            apply_rcut=not is_group,
        )
        w_pol = active_p.to(pd)
        if is_group:
            w_pol = self._group_weights(R_b, pol_k, pol_l, pol_mask, pd)
        elif not self.pbc and self.cutoff_switch == "smooth":
            r_on = max(self.r_cut_bohr - self.switch_width_bohr, 0.0)
            w_pol = w_pol * _quintic_switch(r_p, r_on, self.r_cut_bohr)

        bp = at["beta_pol"].to(pd)
        n_pol = pol_k.shape[0]
        fn_pol = self._eval_pol_coeff_chunk if pol_chunk_fn is None else pol_chunk_fn
        if self.pair_chunk_size is not None and (
            pol_chunk_fn is not None or n_pol > self.pair_chunk_size
        ):
            from torch.utils.checkpoint import checkpoint

            l33_parts, l53_parts = [], []
            step = self.pair_chunk_size
            pad_static = pol_chunk_fn is not None
            for s in range(0, n_pol, step):
                e_idx = min(s + step, n_pol)
                n_loc = e_idx - s
                bk_c = bp[pol_k[s:e_idx]]
                bl_c = bp[pol_l[s:e_idx]]
                rr_c = r_p[s:e_idx]
                wp_c = w_pol[s:e_idx]
                if pad_static:
                    bk_c, bl_c, rr_c, wp_c = self._pad_pol_chunk(
                        bk_c, bl_c, rr_c, wp_c, step
                    )
                c33, c53 = checkpoint(
                    fn_pol,
                    bk_c,
                    bl_c,
                    rr_c,
                    wp_c,
                    use_reentrant=False,
                )
                l33_parts.append(c33[:n_loc])
                l53_parts.append(c53[:n_loc])
            l33_sr = torch.cat(l33_parts)
            l53_sr = torch.cat(l53_parts)
        else:
            l33_sr, l53_sr = fn_pol(bp[pol_k], bp[pol_l], r_p, w_pol)

        S = at["sqrt_alpha"].to(pd)
        rhs = _mat_vec(S, field)
        with torch.no_grad():
            w_star = _sparse_cg_solve(
                S,
                rhs,
                pol_k,
                pol_l,
                rhat_p,
                l33_sr,
                l53_sr,
                self.pol_cg_iters,
                pme=self.elst_pme if self.pbc else None,
                spme_state=spme_state if self.pbc else None,
            )

        p_ind = _mat_vec(S, w_star)
        pk_v, pl_v = p_ind[pol_k], p_ind[pol_l]
        e_pol_pair = (
            l33_sr * _dot(pk_v, pl_v)
            - 3.0 * l53_sr * _dot(pk_v, rhat_p) * _dot(pl_v, rhat_p)
        ).sum()
        e_pol = 0.5 * _dot(w_star, w_star).sum() - _dot(p_ind, field).sum() + e_pol_pair
        if self.pbc:
            assert self.elst_pme is not None and spme_state is not None
            e_pol = e_pol + self.elst_pme.reciprocal_dipole_energy(p_ind, spme_state)

        return torch.stack([elst, exch, e_pol + ct, disp]) * H2KCAL

    def forward(
        self,
        positions_nm: Tensor,
        cell_nm: Tensor | None,
        inter_ai: Tensor,
        inter_aj: Tensor,
        inter_shift: Tensor,
        inter_mask: Tensor,
        pol_k: Tensor,
        pol_l: Tensor,
        pol_shift: Tensor,
        pol_mask: Tensor,
    ) -> Tensor:
        """Compute ``(4,)`` FF-2 components ``[elst, exch, ind, disp]`` in kcal/mol."""
        return self._forward_impl(
            positions_nm,
            cell_nm,
            inter_ai,
            inter_aj,
            inter_shift,
            inter_mask,
            pol_k,
            pol_l,
            pol_shift,
            pol_mask,
        )


class FF2LongRangeEnergy(nn.Module):
    """Fixed-topology FF-2 MD energy (nm -> kJ/mol) with neighbor list and
    optional periodic SPME (multipolar electrostatics, mutual polarization,
    and :math:`r^{-6}` dispersion).

    Parameters
    ----------
    model : FF2Model
        FF-2 model (used as is; :func:`compile_long_range_energy` copies and
        freezes it).
    z, monomer, monomer_charge
        Molecular topology (see :func:`apnet_pt.ff2.md.cluster_batch`).
    r_cut : float
        Real-space cutoff in **Angstrom** (default ``12.0`` Å, matching
        SO3LR-T-M's ``lr_cutoff``).
    skin : float
        Verlet skin buffer in **Angstrom** (default ``1.5`` Å).
    cell : Tensor, shape ``(3, 3)``, optional
        Initial periodic box vectors in **nm** (OpenMM convention). When
        provided (or ``pbc=True``), enables periodic SPME; when ``None``, runs
        non-periodic real-space cutoff + sparse CG polarization.
    pbc : bool, optional
        Explicitly select periodic (``True``) vs cluster (``False``); defaults
        to ``cell is not None``.
    accuracy : float
        Target Ewald/SPME accuracy used by the parameter tuner when
        ``elst_alpha`` / ``elst_mesh`` are not specified (default ``1e-6``).
    elst_alpha : float, optional
        Override electrostatic Ewald :math:`\\alpha` in **bohr^-1**.
    elst_mesh : tuple[int, int, int], optional
        Override electrostatic SPME grid ``(K1, K2, K3)``.
    elst_order : int
        Cardinal B-spline order for multipolar SPME (default ``6``).
    disp_mode : {"pme_lowrank", "pme_geom", "tail", "cutoff"}
        Long-range dispersion treatment in periodic mode (default
        ``"pme_lowrank"`` with rank 4 Chebyshev--Nyström Casimir--Polder).
    disp_rank : int
        Casimir--Polder rank for ``disp_mode="pme_lowrank"`` (default ``4``).
    disp_alpha : float, optional
        Override dispersion Ewald :math:`\\alpha_d` in **bohr^-1**.
    disp_mesh : tuple[int, int, int], optional
        Override dispersion SPME grid ``(K1, K2, K3)``.
    disp_order : int
        B-spline order for dispersion PME (default ``4``).
    c8_tail : bool
        Add homogeneous :math:`C_8` tail correction in periodic mode (default
        ``True`` when ``pbc``).
    pol_cg_iters : int
        Fixed CG iterations for sparse/SPME polarization (default ``20``).
    cutoff_switch : {"group", "smooth", "none"}
        In non-periodic mode, apply a ``C^2`` quintic switch over
        ``[r_cut - switch_width, r_cut]`` on monomer-COM distances (``"group"``,
        default, preserving molecular charge neutrality across the cutoff),
        on atom-atom distances (``"smooth"``), or hard atom cutoff (``"none"``).
        Ignored when ``pbc=True`` (Ewald is already smooth).
    switch_width : float
        Switching window width in **Angstrom** for non-periodic smooth/group
        mode (default ``2.0`` Å).
    pair_chunk_size : int, optional
        If set, evaluate real-space intermolecular and polarization pairs in
        gradient-checkpointed chunks of this size to bound peak activation
        memory to ``O(C) + O(N)``.
    precision : {"fp64", "mixed", "fp32"}, optional
        Precision policy:
        - ``None`` (default): follow ``model.physics_dtype`` everywhere.
        - ``"fp64"``: ``float64`` throughout physics, SPME, and CG.
        - ``"mixed"``: ``float64`` coordinates, displacements, Ewald ``erf``
          subtractions, global reductions, reciprocal SPME, and CG polarization
          around ``float32`` short-range pair kernels (``table2d``, exchange,
          CT, dispersion, dipole tensor) — mirrors SO3LR/OpenMM mixed precision
          to preserve NVE stability at near-``fp32`` speed and VRAM.
        - ``"fp32"``: ``float32`` throughout physics, SPME, and CG (one-mu
          monomer charges always remain ``float64``).
    """

    def __init__(
        self,
        model: FF2Model,
        z: Sequence[int],
        monomer: Sequence[int],
        monomer_charge: Sequence[float],
        r_cut: float = 12.0,
        skin: float = 1.5,
        cell: Tensor | None = None,
        pbc: bool | None = None,
        accuracy: float = 1e-6,
        elst_alpha: float | None = None,
        elst_mesh: tuple[int, int, int] | None = None,
        elst_order: int = 6,
        disp_mode: str = "pme_lowrank",
        disp_rank: int = 4,
        disp_alpha: float | None = None,
        disp_mesh: tuple[int, int, int] | None = None,
        disp_order: int = 4,
        c8_tail: bool = True,
        pol_cg_iters: int = 20,
        cutoff_switch: str = "group",
        switch_width: float = 2.0,
        pair_chunk_size: int | None = None,
        precision: str | None = None,
    ):
        super().__init__()
        self.model = model
        if precision is None:
            state_dtype = model.physics_dtype
            pair_dtype = model.physics_dtype
            self.precision = "fp32" if state_dtype == torch.float32 else "fp64"
        elif precision == "fp64":
            state_dtype, pair_dtype = torch.float64, torch.float64
            self.precision = "fp64"
        elif precision == "mixed":
            state_dtype, pair_dtype = torch.float64, torch.float32
            self.precision = "mixed"
        elif precision == "fp32":
            state_dtype, pair_dtype = torch.float32, torch.float32
            self.precision = "fp32"
        else:
            raise ValueError(
                f"precision must be None, 'fp64', 'mixed', or 'fp32', got {precision!r}"
            )

        self.pbc = (cell is not None) if pbc is None else bool(pbc)
        if self.pbc and cell is None and (elst_alpha is None or elst_mesh is None):
            raise ValueError(
                "periodic mode requires an initial `cell` (nm) to tune SPME "
                "parameters, or explicit `elst_alpha` and `elst_mesh`"
            )

        device = next(model.parameters()).device
        topo = _build_topology_buffers(z, monomer, monomer_charge, device=device)

        self.r_cut_ang = float(r_cut)
        self.r_cut_bohr = self.r_cut_ang * ANG2BOHR
        self.skin_ang = float(skin)
        self.r_cut_nm = self.r_cut_ang / NM2ANG
        self.skin_nm = self.skin_ang / NM2ANG

        elst_pme: MultipolePME | None = None
        disp_pme: DispersionPME | None = None

        if self.pbc:
            cell_b = (
                None
                if cell is None
                else cell.detach().to(dtype=torch.float64, device=device) * NM2BOHR
            )
            if elst_alpha is None or elst_mesh is None:
                assert cell_b is not None
                mp = tune_multipole_pme(
                    cell_b,
                    cutoff=self.r_cut_bohr,
                    accuracy=accuracy,
                    max_rank=2,
                    order=elst_order,
                )
                elst_alpha = mp.alpha if elst_alpha is None else elst_alpha
                elst_mesh = mp.ns_mesh if elst_mesh is None else elst_mesh
            elst_pme = MultipolePME(
                alpha=elst_alpha, ns_mesh=elst_mesh, order=elst_order
            ).to(device)

            if disp_mode in ("pme_lowrank", "pme_geom"):
                if disp_alpha is None or disp_mesh is None:
                    assert cell_b is not None
                    dp = tune_dispersion_pme(
                        cell_b,
                        cutoff=self.r_cut_bohr,
                        accuracy=max(accuracy * 10.0, 1e-6),
                        order=disp_order,
                        rank=disp_rank,
                    )
                    disp_alpha = dp.alpha if disp_alpha is None else disp_alpha
                    disp_mesh = dp.ns_mesh if disp_mesh is None else disp_mesh
                pme_mode = (
                    "geometric" if disp_mode == "pme_geom" else "chebyshev_nystrom"
                )
                disp_pme = DispersionPME(
                    alpha=disp_alpha,
                    ns_mesh=disp_mesh,
                    order=disp_order,
                    rank=disp_rank,
                    mode=pme_mode,
                ).to(device)
            elif disp_mode not in ("tail", "cutoff"):
                raise ValueError(f"unknown disp_mode {disp_mode!r}")

        self.core = _FF2LongRangeCore(
            model=model,
            topo=topo,
            r_cut_bohr=self.r_cut_bohr,
            pbc=self.pbc,
            pol_cg_iters=pol_cg_iters,
            elst_pme=elst_pme,
            disp_pme=disp_pme,
            disp_mode=disp_mode,
            c8_tail=c8_tail and self.pbc,
            cutoff_switch=cutoff_switch,
            switch_width_bohr=float(switch_width) * ANG2BOHR,
            pair_chunk_size=pair_chunk_size,
            state_dtype=state_dtype,
            pair_dtype=pair_dtype,
        )
        self.neighbor_list = NeighborList(
            r_cut=self.r_cut_nm,
            skin=self.skin_nm,
            monomer=topo["monomer"],
            group_cutoff=(not self.pbc) and (cutoff_switch == "group"),
        )
        if cell is not None:
            self.register_buffer(
                "default_cell_nm",
                cell.detach().to(dtype=state_dtype, device=device),
            )
        else:
            self.default_cell_nm = None

    def _resolve_cell(self, cell_nm: Tensor | None, positions_nm: Tensor) -> Tensor | None:
        if not self.pbc:
            return None
        c = self.default_cell_nm if cell_nm is None else cell_nm
        if c is None:
            raise ValueError("cell (nm, shape (3, 3)) is required in periodic mode")
        return c.to(dtype=positions_nm.dtype, device=positions_nm.device)

    def _neighbors(
        self, positions_nm: Tensor, cell_nm: Tensor | None
    ) -> tuple[NeighborBuffers, NeighborBuffers]:
        return self.neighbor_list.update(positions_nm, cell_nm)

    def components(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> Tensor:
        """``(4,)`` ``[elst, exch, ind, disp]`` in kcal/mol."""
        cell = self._resolve_cell(cell_nm, positions_nm)
        inter_buf, all_buf = self._neighbors(positions_nm, cell)
        return self.core(
            positions_nm,
            cell,
            inter_buf.ai,
            inter_buf.aj,
            inter_buf.shift,
            inter_buf.mask,
            all_buf.ai,
            all_buf.aj,
            all_buf.shift,
            all_buf.mask,
        )

    def forward(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> Tensor:
        """Total intermolecular energy in kJ/mol."""
        return self.components(positions_nm, cell_nm).sum() * KCAL2KJ

    def energy_forces(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """``(energy_kj, forces_kj_nm (N, 3))`` at ``positions_nm`` (nm)."""
        device = self.core.topo_z.device
        x = (
            positions_nm.detach()
            .to(device=device, dtype=self.core.state_dtype)
            .requires_grad_(True)
        )
        e = self(x, cell_nm)
        (g,) = torch.autograd.grad(e, x)
        return e.detach(), -g


class CompiledLongRangeEnergy(nn.Module):
    """Inductor-compiled :class:`FF2LongRangeEnergy` with zero recompiles
    across neighbor-list rebuilds of the same static capacity.

    When ``module.core.pair_chunk_size`` is set, compiles the fixed-size
    intermolecular and polarization chunk kernels once (static shape
    ``(pair_chunk_size,)``) and runs the gradient-checkpointed chunk loop
    outside ``torch.compile`` so peak activation memory stays ``O(C) + O(N)``
    while every chunk executes fused Inductor kernels with constant compile
    time independent of system size.
    """

    def __init__(self, module: FF2LongRangeEnergy, **options):
        super().__init__()
        self.module = module
        opts = {"fullgraph": True, "dynamic": False, **options}
        if module.core.pair_chunk_size is not None:
            self._compiled_core = None
            self._compiled_inter_chunk = torch.compile(
                module.core._eval_inter_chunk, **opts
            )
            self._compiled_pol_chunk = torch.compile(
                module.core._eval_pol_coeff_chunk, **opts
            )
        else:
            self._compiled_core = torch.compile(module.core, **opts)
            self._compiled_inter_chunk = None
            self._compiled_pol_chunk = None

    def components(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> Tensor:
        cell = self.module._resolve_cell(cell_nm, positions_nm)
        inter_buf, all_buf = self.module._neighbors(positions_nm, cell)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            if self._compiled_core is not None:
                return self._compiled_core(
                    positions_nm,
                    cell,
                    inter_buf.ai,
                    inter_buf.aj,
                    inter_buf.shift,
                    inter_buf.mask,
                    all_buf.ai,
                    all_buf.aj,
                    all_buf.shift,
                    all_buf.mask,
                )
            return self.module.core._forward_impl(
                positions_nm,
                cell,
                inter_buf.ai,
                inter_buf.aj,
                inter_buf.shift,
                inter_buf.mask,
                all_buf.ai,
                all_buf.aj,
                all_buf.shift,
                all_buf.mask,
                inter_chunk_fn=self._compiled_inter_chunk,
                pol_chunk_fn=self._compiled_pol_chunk,
            )

    def forward(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> Tensor:
        return self.components(positions_nm, cell_nm).sum() * KCAL2KJ

    def energy_forces(
        self, positions_nm: Tensor, cell_nm: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        device = self.module.core.topo_z.device
        x = (
            positions_nm.detach()
            .to(device=device, dtype=self.module.core.state_dtype)
            .requires_grad_(True)
        )
        e = self(x, cell_nm)
        (g,) = torch.autograd.grad(e, x)
        return e.detach(), -g


def compile_long_range_energy(
    model: FF2Model,
    z: Sequence[int],
    monomer: Sequence[int],
    monomer_charge: Sequence[float],
    positions_nm: Tensor,
    cell_nm: Tensor | None = None,
    **kwargs,
) -> CompiledLongRangeEnergy:
    """Build and compile :class:`FF2LongRangeEnergy` for fixed topology.

    Any keyword arguments for ``torch.compile`` may be passed via
    ``compile_options={...}``; all other keyword arguments go to
    :class:`FF2LongRangeEnergy`.
    """
    compile_options = kwargs.pop("compile_options", {})
    frozen = copy.deepcopy(model).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    prec = kwargs.get("precision")
    if prec in ("fp64", "mixed"):
        frozen.physics_dtype = torch.float64
    elif prec == "fp32":
        frozen.physics_dtype = torch.float32
    module = FF2LongRangeEnergy(
        frozen, z, monomer, monomer_charge, cell=cell_nm, **kwargs
    )
    device = module.core.topo_z.device
    pos_w = positions_nm.detach().to(device=device, dtype=module.core.state_dtype)
    # Warm up kernel tables & neighbor-list buffers eagerly before compiling
    with torch.no_grad():
        module(pos_w, cell_nm)
    return CompiledLongRangeEnergy(module, **compile_options)


def openmm_python_force(
    module: FF2LongRangeEnergy | CompiledLongRangeEnergy,
    force_group: int = 0,
    name: str = "FF2_PME",
):
    """Wrap an FF-2 long-range energy module in OpenMM 8.5's ``PythonForce``.

    Passes both atomic positions (nm) and, when ``module`` is periodic, the
    step's periodic box vectors (nm) to ``module.energy_forces``.
    """
    import numpy as np
    import openmm as mm

    nm = mm.unit.nanometer
    is_pbc = (
        module.module.pbc
        if isinstance(module, CompiledLongRangeEnergy)
        else module.pbc
    )
    device = next(module.parameters()).device

    def compute(state):
        pos_np = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(nm), dtype=np.float64
        )
        pos_t = torch.from_numpy(pos_np).to(device=device)
        cell_t = None
        if is_pbc:
            box_np = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nm),
                dtype=np.float64,
            )
            cell_t = torch.from_numpy(box_np).to(device=device)
        e, f = module.energy_forces(pos_t, cell_t)
        return float(e.cpu()), f.detach().cpu().numpy()

    force = mm.PythonForce(compute)
    force.setUsesPeriodicBoundaryConditions(is_pbc)
    force.setForceGroup(force_group)
    force.setName(name)
    return force


__all__ = [
    "CompiledLongRangeEnergy",
    "FF2LongRangeEnergy",
    "NM2BOHR",
    "compile_long_range_energy",
    "openmm_python_force",
]
