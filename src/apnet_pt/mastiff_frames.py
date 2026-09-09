"""Permutation-compatible averaging of explicitly assigned MASTIFF body frames.

Chemical reference assignment remains an external, audited preprocessing step.
This module differentiates geometry and vectorizes the resulting frame plan.
"""

import torch
from torch import Tensor, nn

from .mastiff import RacahHarmonics, local_frames


class AveragedFrameHarmonics(nn.Module):
    """Average equivalent body-frame features before multiplying pair factors.

    Full frames retain cosine channels 10,11c,20,21c,22c. Sine channels are
    suppressed to preserve reflection invariance with scalar graph-based
    coefficients and right-handed geometrical frames. This is a restricted
    angular model, not a claim that every atom has local mirror symmetry.
    Axial sites use only 10,20. Sites in neither plan are explicitly isotropic.
    Reference plan indices must be assigned independently of atom numbering.
    """

    def __init__(self, backend: str = "torch", method: str | None = None):
        super().__init__()
        self.harmonics = RacahHarmonics(backend, method)
        self.register_buffer("cosine_mask", torch.tensor([1, 1, 0, 1, 1, 0, 1, 0]))

    def forward(
        self,
        positions: Tensor,
        directions: Tensor,
        full: Tensor,
        axial: Tensor,
        validate: bool = True,
    ) -> Tensor:
        """Return (atoms, partners, 8) atomic angular features.

        full is (frames,3) with columns (center,z_reference,x_reference).
        axial is (frames,2) with columns (center,z_reference). Multiple rows
        per center are averaged with equal weight. A center must not appear
        in both plans. Integer plans may be empty with their stated shapes.
        Directions are outgoing center-to-partner vectors in laboratory axes.
        Coordinates and directions must use the same length unit.
        """
        if validate:
            if bool(torch.any(full < 0) | torch.any(full >= positions.shape[0])):
                raise ValueError("full-frame index outside monomer")
            if bool(torch.any(axial < 0) | torch.any(axial >= positions.shape[0])):
                raise ValueError("axial index outside monomer")
            if bool(torch.isin(full[:, 0], axial[:, 0]).any()):
                raise ValueError("site cannot be both full-frame and axial")
            if not bool(torch.isfinite(directions).all()) or bool(
                (directions.norm(dim=-1) == 0).any()
            ):
                raise ValueError("nonfinite or zero pair direction")
        output = directions.new_zeros((*directions.shape[:-1], 8))
        counts = directions.new_zeros(positions.shape[0])
        center, iz, ix = full.unbind(-1)
        frames = local_frames(
            positions[iz] - positions[center],
            positions[ix] - positions[center],
            validate=validate,
        )
        local = torch.einsum("fpi,fij->fpj", directions[center], frames)
        values = self.harmonics(local) * self.cosine_mask
        output = output.index_add(0, center, values)
        counts = counts.index_add(
            0, center, torch.ones_like(center, dtype=positions.dtype)
        )
        ac, az = axial.unbind(-1)
        z = positions[az] - positions[ac]
        norms = z.norm(dim=-1, keepdim=True)
        if validate and bool((~torch.isfinite(norms) | (norms <= 1e-10)).any()):
            raise ValueError("degenerate axial reference")
        z = z / norms
        unit = directions[ac] / directions[ac].norm(dim=-1, keepdim=True)
        cosine = (unit * z[:, None]).sum(-1)
        zero = torch.zeros_like(cosine)
        values = torch.stack(
            (cosine, zero, zero, (3 * cosine.square() - 1) / 2, zero, zero, zero, zero),
            -1,
        )
        output = output.index_add(0, ac, values)
        counts = counts.index_add(0, ac, torch.ones_like(ac, dtype=positions.dtype))
        return output / counts.clamp_min(1)[:, None, None]
