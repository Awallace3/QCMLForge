"""Compile-friendly cell-linked neighbor list with static capacity and skin.

Supports both periodic (orthorhombic and arbitrary triclinic cells, including
small boxes where ``r_list >= L / 2``) and non-periodic cluster systems.
Between rebuilds, the neighbor tensors keep identical shapes so
``torch.compile(fullgraph=True, dynamic=False)`` executes without recompiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NeighborBuffers:
    """Static-shape padded half-list of atom pairs ``(i, j, shift, mask)``.

    For every active entry (``mask == 1``), the pair vector from ``i`` to ``j``
    is ``positions[j] - positions[i] + shift @ cell`` (with ``shift = 0`` for
    non-periodic systems).  Padded entries have ``i = j = 0``, ``shift = 0``,
    and ``mask = 0``.
    """

    ai: Tensor  # (C,) int64
    aj: Tensor  # (C,) int64
    shift: Tensor  # (C, 3) float (integer lattice shifts in box basis)
    mask: Tensor  # (C,) bool
    n_valid: int


# 13 forward half-stencil neighbor cell offsets in 3D (excluding (0, 0, 0))
_HALF_OFFSETS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)


def _pad_pairs(
    ai: Tensor,
    aj: Tensor,
    shift: Tensor,
    capacity: int,
    dtype: torch.dtype,
    device: torch.device,
) -> NeighborBuffers:
    n_valid = int(ai.shape[0])
    cap = max(capacity, n_valid, 1)
    pad = cap - n_valid
    if pad > 0:
        ai = torch.cat([ai, torch.zeros(pad, dtype=torch.long, device=device)])
        aj = torch.cat([aj, torch.zeros(pad, dtype=torch.long, device=device)])
        shift = torch.cat(
            [shift.to(dtype), torch.zeros((pad, 3), dtype=dtype, device=device)]
        )
        mask = torch.cat(
            [
                torch.ones(n_valid, dtype=torch.bool, device=device),
                torch.zeros(pad, dtype=torch.bool, device=device),
            ]
        )
    else:
        shift = shift.to(dtype)
        mask = torch.ones(n_valid, dtype=torch.bool, device=device)
    return NeighborBuffers(ai=ai, aj=aj, shift=shift, mask=mask, n_valid=n_valid)


def _build_allpairs_nonperiodic(
    positions: Tensor,
    r_list: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Direct O(N^2) half-list for small non-periodic clusters."""
    n = positions.shape[0]
    device = positions.device
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    diff = positions[jj] - positions[ii]
    r2 = (diff * diff).sum(dim=-1)
    keep = r2 < (r_list * r_list)
    ai, aj = ii[keep], jj[keep]
    shift = torch.zeros((ai.shape[0], 3), dtype=positions.dtype, device=device)
    return ai, aj, shift


def _build_allpairs_periodic(
    positions: Tensor,
    cell: Tensor,
    r_list: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact image-enumerated half-list for small periodic boxes (arbitrary
    triclinic cells, even when ``r_list >= L / 2``)."""
    dtype, device = positions.dtype, positions.device
    n = positions.shape[0]
    inv_cell = torch.linalg.inv(cell)
    # Wrap atoms into primary cell [0, 1)^3 for clean minimum-image search
    frac = positions @ inv_cell
    wrap = torch.floor(frac)
    frac_w = frac - wrap
    pos_w = frac_w @ cell

    heights = 1.0 / torch.linalg.norm(inv_cell, dim=0)
    n_img = [int(math.ceil(r_list / float(h))) + 1 for h in heights]

    # Build half-space of lattice shifts: (0,0,0) with i<j, plus positive half-space
    # (first nonzero entry > 0) with ALL (i, j) including i == j!
    s_list = []
    for sx in range(-n_img[0], n_img[0] + 1):
        for sy in range(-n_img[1], n_img[1] + 1):
            for sz in range(-n_img[2], n_img[2] + 1):
                if sx > 0 or (sx == 0 and sy > 0) or (sx == 0 and sy == 0 and sz > 0):
                    s_list.append((sx, sy, sz))

    r2_cut = r_list * r_list
    ai_parts, aj_parts, sh_parts = [], [], []

    # (0, 0, 0) shift: i < j
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    diff0 = pos_w[jj] - pos_w[ii]
    keep0 = (diff0 * diff0).sum(dim=-1) < r2_cut
    if bool(keep0.any()):
        pi, pj = ii[keep0], jj[keep0]
        # True shift for unwrapped positions: pos[j] - pos[i] + (wrap[i] - wrap[j]) @ cell
        sh0 = wrap[pi] - wrap[pj]
        ai_parts.append(pi)
        aj_parts.append(pj)
        sh_parts.append(sh0)

    # Positive half-space shifts: all (i, j) pairs
    idx = torch.arange(n, device=device)
    fi, fj = torch.meshgrid(idx, idx, indexing="ij")
    fi, fj = fi.reshape(-1), fj.reshape(-1)
    base_diff = pos_w[fj] - pos_w[fi]
    base_wrap = wrap[fi] - wrap[fj]

    for sx, sy, sz in s_list:
        s_vec = torch.tensor([sx, sy, sz], dtype=dtype, device=device)
        d = base_diff + (s_vec @ cell)
        keep = (d * d).sum(dim=-1) < r2_cut
        if bool(keep.any()):
            ai_parts.append(fi[keep])
            aj_parts.append(fj[keep])
            sh_parts.append(base_wrap[keep] + s_vec)

    if not ai_parts:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        empty_s = torch.zeros((0, 3), dtype=dtype, device=device)
        return empty_i, empty_i, empty_s

    return torch.cat(ai_parts), torch.cat(aj_parts), torch.cat(sh_parts)


def _build_cell_list(
    positions: Tensor,
    cell: Tensor | None,
    r_list: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized O(N) 3D cell-linked half-list for periodic or non-periodic
    systems."""
    dtype, device = positions.dtype, positions.device
    n = positions.shape[0]
    if n < 2:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return empty_i, empty_i, torch.zeros((0, 3), dtype=dtype, device=device)

    periodic = cell is not None
    if periodic:
        assert cell is not None
        inv_cell = torch.linalg.inv(cell)
        heights = 1.0 / torch.linalg.norm(inv_cell, dim=0)
        nc = [int(math.floor(float(h) / r_list)) for h in heights]
        if min(nc) < 3:
            return _build_allpairs_periodic(positions, cell, r_list)
        frac = positions @ inv_cell
        wrap = torch.floor(frac)
        frac_w = frac - wrap
        pos_w = frac_w @ cell
        cx = torch.clamp((frac_w[:, 0] * nc[0]).long(), 0, nc[0] - 1)
        cy = torch.clamp((frac_w[:, 1] * nc[1]).long(), 0, nc[1] - 1)
        cz = torch.clamp((frac_w[:, 2] * nc[2]).long(), 0, nc[2] - 1)
    else:
        if n <= 384:
            return _build_allpairs_nonperiodic(positions, r_list)
        p_min = positions.min(dim=0).values
        p_max = positions.max(dim=0).values
        span = (p_max - p_min).clamp_min(r_list)
        nc = [max(1, int(math.floor(float(s) / r_list))) for s in span]
        if nc[0] * nc[1] * nc[2] < 27:
            return _build_allpairs_nonperiodic(positions, r_list)
        rel = positions - p_min
        cx = torch.clamp((rel[:, 0] / r_list).long(), 0, nc[0] - 1)
        cy = torch.clamp((rel[:, 1] / r_list).long(), 0, nc[1] - 1)
        cz = torch.clamp((rel[:, 2] / r_list).long(), 0, nc[2] - 1)
        pos_w = positions
        wrap = torch.zeros_like(positions)

    nx, ny, nz = nc
    n_cells = nx * ny * nz
    cell_id = (cx * ny + cy) * nz + cz

    # Pack atoms into padded (n_cells, max_occ) array using vectorized argsort
    order = torch.argsort(cell_id)
    sorted_cid = cell_id[order]
    counts = torch.bincount(sorted_cid, minlength=n_cells)
    max_occ = int(counts.max().item())
    if max_occ == 0:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return empty_i, empty_i, torch.zeros((0, 3), dtype=dtype, device=device)

    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(n, device=device) - starts[sorted_cid]
    cell_atoms = torch.full((n_cells, max_occ), -1, dtype=torch.long, device=device)
    cell_atoms[sorted_cid, slot] = order

    # Only iterate over non-empty source cells
    nonempty = torch.nonzero(counts > 0, as_tuple=True)[0]
    src_atoms = cell_atoms[nonempty]  # (M, K)
    src_valid = src_atoms >= 0
    src_safe = src_atoms.clamp_min(0)
    sx0 = nonempty // (ny * nz)
    sy0 = (nonempty // nz) % ny
    sz0 = nonempty % nz

    r2_cut = r_list * r_list
    ai_parts, aj_parts, sh_parts = [], [], []

    # 1. Self-cell pairs (upper triangle a < b)
    a_idx, b_idx = torch.triu_indices(max_occ, max_occ, offset=1, device=device)
    if a_idx.numel() > 0:
        valid_self = src_valid[:, a_idx] & src_valid[:, b_idx]
        pi = src_safe[:, a_idx][valid_self]
        pj = src_safe[:, b_idx][valid_self]
        d = pos_w[pj] - pos_w[pi]
        keep = (d * d).sum(dim=-1) < r2_cut
        if bool(keep.any()):
            pi, pj = pi[keep], pj[keep]
            ai_parts.append(pi)
            aj_parts.append(pj)
            sh_parts.append(wrap[pi] - wrap[pj])

    # 2. 13 forward neighbor cells
    for dx, dy, dz in _HALF_OFFSETS:
        tx_raw = sx0 + dx
        ty_raw = sy0 + dy
        tz_raw = sz0 + dz
        if periodic:
            assert cell is not None
            img_x = torch.div(tx_raw, nx, rounding_mode="floor")
            img_y = torch.div(ty_raw, ny, rounding_mode="floor")
            img_z = torch.div(tz_raw, nz, rounding_mode="floor")
            tx = tx_raw - img_x * nx
            ty = ty_raw - img_y * ny
            tz = tz_raw - img_z * nz
            dst_cid = (tx * ny + ty) * nz + tz
            dst_atoms = cell_atoms[dst_cid]  # (M, K)
            dst_valid = dst_atoms >= 0
            valid = src_valid[:, :, None] & dst_valid[:, None, :]
            if not bool(valid.any()):
                continue
            cell_idx, a_loc, b_loc = torch.nonzero(valid, as_tuple=True)
            pi = src_atoms[cell_idx, a_loc]
            pj = dst_atoms[cell_idx, b_loc]
            img = torch.stack(
                [img_x[cell_idx], img_y[cell_idx], img_z[cell_idx]], dim=-1
            ).to(dtype)
            d = pos_w[pj] - pos_w[pi] + img @ cell
            keep = (d * d).sum(dim=-1) < r2_cut
            if bool(keep.any()):
                pi, pj, img = pi[keep], pj[keep], img[keep]
                ai_parts.append(pi)
                aj_parts.append(pj)
                sh_parts.append(wrap[pi] - wrap[pj] + img)
        else:
            in_bounds = (
                (tx_raw >= 0)
                & (tx_raw < nx)
                & (ty_raw >= 0)
                & (ty_raw < ny)
                & (tz_raw >= 0)
                & (tz_raw < nz)
            )
            if not bool(in_bounds.any()):
                continue
            dst_cid = (tx_raw[in_bounds] * ny + ty_raw[in_bounds]) * nz + tz_raw[
                in_bounds
            ]
            s_a = src_atoms[in_bounds]
            s_v = src_valid[in_bounds]
            d_a = cell_atoms[dst_cid]
            d_v = d_a >= 0
            valid = s_v[:, :, None] & d_v[:, None, :]
            if not bool(valid.any()):
                continue
            cell_idx, a_loc, b_loc = torch.nonzero(valid, as_tuple=True)
            pi = s_a[cell_idx, a_loc]
            pj = d_a[cell_idx, b_loc]
            d = pos_w[pj] - pos_w[pi]
            keep = (d * d).sum(dim=-1) < r2_cut
            if bool(keep.any()):
                pi, pj = pi[keep], pj[keep]
                ai_parts.append(pi)
                aj_parts.append(pj)
                sh_parts.append(torch.zeros((pi.shape[0], 3), dtype=dtype, device=device))

    if not ai_parts:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return empty_i, empty_i, torch.zeros((0, 3), dtype=dtype, device=device)
    return torch.cat(ai_parts), torch.cat(aj_parts), torch.cat(sh_parts)


class NeighborList:
    """Skin-buffered, fixed-capacity neighbor list manager for MD.

    Maintains separate static-capacity half-lists for:

    * **all pairs** within ``r_cut + skin`` (used for mutual polarization);
    * **intermolecular pairs** (``monomer[i] != monomer[j]``) within
      ``r_cut + skin`` (used for electrostatics, permanent field, exchange,
      charge transfer, and dispersion).

    Rebuilds only when any atom moves by more than ``skin / 2`` or when the
    box vectors change by more than ``cell_tol``.

    Parameters
    ----------
    r_cut : float
        Interaction cutoff distance (in the same length unit as ``positions``).
    skin : float
        Verlet skin buffer (default ``0.0`` rebuilds every call; typical MD
        value is ``1.0``–``2.0`` bohr or ``0.5``–``1.0`` Å).
    monomer : Tensor, shape ``(N,)``, optional
        Monomer index of each atom. If provided, separates intermolecular
        pairs from all-atom pairs.
    capacity_factor : float
        Headroom multiplier on initial/resized capacity (default ``1.25``) to
        avoid shape changes across MD steps.
    cell_tol : float
        Rebuild if ``max |cell - ref_cell| > cell_tol``.
    """

    def __init__(
        self,
        r_cut: float,
        skin: float = 0.0,
        monomer: Tensor | None = None,
        capacity_factor: float = 1.25,
        cell_tol: float = 1e-6,
        group_cutoff: bool = False,
    ):
        if r_cut <= 0:
            raise ValueError(f"r_cut must be positive, got {r_cut}")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")
        if group_cutoff and monomer is None:
            raise ValueError("group_cutoff=True requires `monomer`")
        self.r_cut = float(r_cut)
        self.skin = float(skin)
        self.r_list = self.r_cut + self.skin
        self.monomer = monomer
        self.capacity_factor = float(capacity_factor)
        self.cell_tol = float(cell_tol)
        self.group_cutoff = bool(group_cutoff)

        self._ref_positions: Tensor | None = None
        self._ref_cell: Tensor | None = None
        self._all_cap: int = 0
        self._inter_cap: int = 0
        self._all_buf: NeighborBuffers | None = None
        self._inter_buf: NeighborBuffers | None = None
        self.n_rebuilds: int = 0

    def needs_rebuild(self, positions: Tensor, cell: Tensor | None) -> bool:
        if self._ref_positions is None or self._all_buf is None:
            return True
        if self.skin == 0.0:
            return True
        if (cell is None) != (self._ref_cell is None):
            return True
        if cell is not None and self._ref_cell is not None:
            if float((cell.detach() - self._ref_cell).abs().max().item()) > self.cell_tol:
                return True
        max_disp2 = float(
            ((positions.detach() - self._ref_positions) ** 2).sum(dim=-1).max().item()
        )
        return max_disp2 > (0.5 * self.skin) ** 2

    def update(
        self,
        positions: Tensor,
        cell: Tensor | None = None,
        force: bool = False,
    ) -> tuple[NeighborBuffers, NeighborBuffers]:
        """Return ``(inter_buf, all_buf)``, rebuilding only if needed.

        When ``monomer`` was not supplied, ``inter_buf`` is identical to
        ``all_buf``.
        """
        if not force and not self.needs_rebuild(positions, cell):
            assert self._inter_buf is not None and self._all_buf is not None
            return self._inter_buf, self._all_buf

        pos_d = positions.detach()
        cell_d = None if cell is None else cell.detach()
        dtype, device = pos_d.dtype, pos_d.device

        if self.group_cutoff and cell_d is None and self.monomer is not None:
            # Non-periodic charge-group (monomer-COM) neighbor list: keeps
            # every monomer electrically neutral across the cutoff.
            mon = self.monomer.to(device=device)
            n_mon = int(mon.max().item()) + 1
            counts = torch.bincount(mon, minlength=n_mon)
            coms = pos_d.new_zeros((n_mon, 3)).index_add(0, mon, pos_d) / counts[
                :, None
            ].to(dtype)
            ma, mb, _ = _build_cell_list(coms, None, self.r_list)

            # Group atom indices by monomer
            order = torch.argsort(mon, stable=True)
            starts = torch.cumsum(counts, dim=0) - counts
            max_m = int(counts.max().item())
            m_atoms = torch.full((n_mon, max_m), -1, dtype=torch.long, device=device)
            slot = torch.arange(pos_d.shape[0], device=device) - starts[mon[order]]
            m_atoms[mon[order], slot] = order

            # Expand monomer pairs (ma, mb) to intermolecular atom pairs
            if ma.numel() > 0:
                a_grid = m_atoms[ma]  # (P_m, max_m)
                b_grid = m_atoms[mb]  # (P_m, max_m)
                valid_ab = (a_grid[:, :, None] >= 0) & (b_grid[:, None, :] >= 0)
                p_idx, la, lb = torch.nonzero(valid_ab, as_tuple=True)
                ai_i = a_grid[p_idx, la]
                aj_i = b_grid[p_idx, lb]
                sh_i = torch.zeros((ai_i.shape[0], 3), dtype=dtype, device=device)
            else:
                ai_i = torch.zeros(0, dtype=torch.long, device=device)
                aj_i = ai_i
                sh_i = torch.zeros((0, 3), dtype=dtype, device=device)

            # Intramonomer atom pairs (for polarization all-atom list)
            ii_m, jj_m = torch.triu_indices(max_m, max_m, offset=1, device=device)
            if ii_m.numel() > 0:
                valid_in = (m_atoms[:, ii_m] >= 0) & (m_atoms[:, jj_m] >= 0)
                in_i = m_atoms[:, ii_m][valid_in]
                in_j = m_atoms[:, jj_m][valid_in]
                ai = torch.cat([ai_i, in_i])
                aj = torch.cat([aj_i, in_j])
            else:
                ai, aj = ai_i, aj_i
            shift = torch.zeros((ai.shape[0], 3), dtype=dtype, device=device)

            n_all = int(ai.shape[0])
            if n_all > self._all_cap or self._all_cap == 0:
                self._all_cap = max(int(math.ceil(n_all * self.capacity_factor)), 1)
            self._all_buf = _pad_pairs(ai, aj, shift, self._all_cap, dtype, device)

            n_inter = int(ai_i.shape[0])
            if n_inter > self._inter_cap or self._inter_cap == 0:
                self._inter_cap = max(
                    int(math.ceil(n_inter * self.capacity_factor)), 1
                )
            self._inter_buf = _pad_pairs(
                ai_i, aj_i, sh_i, self._inter_cap, dtype, device
            )
            self._ref_positions = pos_d.clone()
            self._ref_cell = None
            self.n_rebuilds += 1
            return self._inter_buf, self._all_buf

        ai, aj, shift = _build_cell_list(pos_d, cell_d, self.r_list)

        n_all = int(ai.shape[0])
        if n_all > self._all_cap or self._all_cap == 0:
            self._all_cap = max(int(math.ceil(n_all * self.capacity_factor)), 1)
        self._all_buf = _pad_pairs(ai, aj, shift, self._all_cap, dtype, device)

        if self.monomer is not None:
            mon = self.monomer.to(device=device)
            same_mon = mon[ai] == mon[aj]
            if cell_d is not None and bool(same_mon.any()):
                # For same-monomer pairs in PBC, exclude ONLY the minimum-image
                # (intramonomer) copy, keeping genuine periodic-image copies
                # even when a molecule is atom-wrapped across box boundaries.
                inv_cell = torch.linalg.inv(cell_d)
                frac_d = (pos_d[aj[same_mon]] - pos_d[ai[same_mon]]) @ inv_cell
                mic_shift = -torch.round(frac_d)
                is_image_copy = (
                    (ai[same_mon] == aj[same_mon])
                    | ((shift[same_mon] - mic_shift).abs() > 0.5).any(dim=-1)
                )
                is_inter = ~same_mon
                is_inter[same_mon] = is_image_copy
            else:
                is_inter = ~same_mon
            ai_i, aj_i, sh_i = ai[is_inter], aj[is_inter], shift[is_inter]
            n_inter = int(ai_i.shape[0])
            if n_inter > self._inter_cap or self._inter_cap == 0:
                self._inter_cap = max(int(math.ceil(n_inter * self.capacity_factor)), 1)
            self._inter_buf = _pad_pairs(
                ai_i, aj_i, sh_i, self._inter_cap, dtype, device
            )
        else:
            self._inter_buf = self._all_buf

        self._ref_positions = pos_d.clone()
        self._ref_cell = None if cell_d is None else cell_d.clone()
        self.n_rebuilds += 1
        return self._inter_buf, self._all_buf


__all__ = ["NeighborBuffers", "NeighborList"]
