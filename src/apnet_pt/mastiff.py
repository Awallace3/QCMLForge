"""Explicit MASTIFF exchange (J. Phys. Chem. A 2023, 127, 1736, Eqs. 2–3).

This opt-in module does not alter CLIFF or AP3 checkpoint semantics. Coordinates
and exponents must use reciprocal units; output units are those of A_i * A_j.
The real Racah basis is (10, 11c, 11s, 20, 21c, 21s, 22c, 22s), with
C_lm = sqrt(4*pi/(2*l+1)) Y_lm and the CANplugin positive Cartesian convention.
"""

import math

import torch
from torch import Tensor, nn

HARMONIC_LABELS = ('10', '11c', '11s', '20', '21c', '21s', '22c', '22s')
SYMMETRY_CHANNELS = {
    'general': tuple(range(8)),
    'c2v': (0, 3, 6),
    'axial': (0, 3),
    'isotropic': (),
}


class RacahHarmonics(nn.Module):
    """Evaluate all real l=1,2 harmonics on nonzero Cartesian vectors.

    Parameters
    ----------
    backend : {'torch', 'cuequivariance'}
        Torch uses exact Cartesian solid harmonics evaluated on unit vectors,
        not a multipole proxy. cuEquivariance is an optional accelerated backend.
    method : str or None
        cuEquivariance kernel selection; use 'naive' for CPU/reference execution.
        None selects the library default CUDA implementation.
    """

    def __init__(self, backend: str = 'torch', method: str | None = None):
        super().__init__()
        if backend not in ('torch', 'cuequivariance'):
            raise ValueError('backend must be torch or cuequivariance')
        self.backend = backend
        if backend == 'cuequivariance':
            try:
                from cuequivariance_torch import SphericalHarmonics
            except ImportError as exc:
                raise ImportError('Install qcmlforge[mastiff] for this backend') from exc
            self.harmonics = SphericalHarmonics([1, 2], normalize=True, method=method)
        self.register_buffer('order', torch.tensor([1, 2, 0, 5, 6, 4, 7, 3]))
        # Integer degree buffer avoids rounding constants before .double().
        self.register_buffer('degrees', torch.tensor([3, 3, 3, 5, 5, 5, 5, 5]))

    def forward(self, vectors: Tensor) -> Tensor:
        """Return (..., 8) harmonics; zero vectors are outside the domain."""
        if self.backend == 'cuequivariance':
            # cuEquivariance uses the e3nn real basis with polar axis y.
            # Cyclic (y,z,x) input gives the conventional polar-z CAN basis.
            v = vectors[..., [1, 2, 0]]
            values = self.harmonics(v.reshape(-1, 3)).reshape(*v.shape[:-1], 8)
            return values.index_select(-1, self.order) / self.degrees.to(values).sqrt()
        unit = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        x, y, z = unit.unbind(-1)
        root3 = math.sqrt(3)
        return torch.stack((z, x, y, (3 * z.square() - 1) / 2,
                            root3 * x * z, root3 * y * z,
                            root3 / 2 * (x.square() - y.square()),
                            root3 * x * y), dim=-1)


def _unit(vector: Tensor, validate: bool, tolerance: float) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if validate and bool(torch.any(~torch.isfinite(norm) | (norm <= tolerance))):
        raise ValueError('degenerate local frame: zero, collinear or nonfinite axes')
    return vector / norm


def local_frames(
    z_reference: Tensor,
    x_reference: Tensor | None = None,
    y_reference: Tensor | None = None,
    *,
    kind: str = 'z-then-x',
    validate: bool = True,
    tolerance: float = 1e-10,
) -> Tensor:
    """Construct right-handed CAN-style body frames with columns (x,y,z).

    Inputs are reference-atom minus central-atom displacement vectors, (...,3),
    in a consistent coordinate unit. Topology/reference selection happens outside
    this function and must not change with atom numbering or partner geometry.
    For 'bisector', x_reference is CAN's second bisected bond (AtomY), not AtomX.
    'z-bisect' and 'threefold' require all three reference vectors.

    Raw displacements, NOT normalized bonds, are averaged as in CAN. Bisector
    X is normal to the reference plane. 'z-only' and 'threefold' supply an
    arbitrary transverse gauge: use ONLY axial coefficients (10,20). A full non-axial frame cannot be inferred from one
    bond. validate=False avoids host synchronization for torch.compile, but is
    only safe for prevalidated, nondegenerate geometries. Degeneracy is never
    repaired with a lab-fixed fallback for non-axial sites.
    """
    kinds = ('z-then-x', 'bisector', 'z-bisect', 'threefold', 'z-only')
    if kind not in kinds:
        raise ValueError(f'kind must be one of {kinds}')
    z = z_reference
    x = x_reference
    if kind != 'z-only' and x is None:
        raise ValueError(f'{kind} requires x_reference')
    if validate:
        _unit(z_reference, True, tolerance)
        if kind != 'z-only':
            _unit(x_reference, True, tolerance)
        if kind in ('z-bisect', 'threefold') and y_reference is not None:
            _unit(y_reference, True, tolerance)
    if kind == 'bisector':
        # CAN uses raw bonds Z and Y, with transverse X normal to their plane.
        x = torch.linalg.cross(z_reference, x_reference, dim=-1)
        z = (z_reference + x_reference) / 2
    elif kind in ('z-bisect', 'threefold'):
        if y_reference is None:
            raise ValueError(f'{kind} requires y_reference')
        if kind == 'z-bisect':
            x = (x_reference + y_reference) / 2
        else:
            z = (z_reference + x_reference + y_reference) / 3
    z = _unit(z, validate, tolerance)
    if kind in ('z-only', 'threefold'):
        # CAN only supplies a chemically meaningful Z for these modes. Use a
        # nonsingular gauge, not its length-dependent lab-axis threshold.
        # This gauge is unobservable ONLY when m!=0 coefficients are masked.
        index = z.abs().argmin(dim=-1)
        x = torch.nn.functional.one_hot(index, 3).to(z)
    x = _unit(x - (x * z).sum(-1, keepdim=True) * z, validate, tolerance)
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _symmetry_mask(symmetry: str) -> Tensor:
    if symmetry not in SYMMETRY_CHANNELS:
        raise ValueError(f'unknown site symmetry: {symmetry}')
    mask = torch.zeros(8, dtype=torch.bool)
    mask[list(SYMMETRY_CHANNELS[symmetry])] = True
    return mask


class MastiffExchange(nn.Module):
    """Per-edge MASTIFF exchange with free atomwise SH coefficients.

    The literal model is A_i*A_j*(1+sum a_i C_i)*(1+sum a_j C_j)
    * (1+x+x^2/3)*exp(-x), x=sqrt(B_i*B_j)*r. No exponential angular
    transform, exponent anisotropy, width clamp or energy conversion is hidden
    here. In particular the linear factors are NOT guaranteed positive.

    Inputs may be gathered from per-atom parameter tables or MPNN readouts;
    no atom types, base checkpoints, or electrostatic multipoles are required.
    Use masks for chemical symmetry, not atom-index-based frame choices.
    """

    def __init__(self, backend: str = 'torch', method: str | None = None,
                 symmetry_i: str = 'general', symmetry_j: str = 'general'):
        super().__init__()
        self.harmonics = RacahHarmonics(backend, method)
        self.register_buffer('mask_i', _symmetry_mask(symmetry_i))
        self.register_buffer('mask_j', _symmetry_mask(symmetry_j))

    def forward(self, displacement: Tensor, a_i: Tensor, a_j: Tensor,
                b_i: Tensor, b_j: Tensor, coefficients_i: Tensor,
                coefficients_j: Tensor, frame_i: Tensor, frame_j: Tensor) -> Tensor:
        """Evaluate aligned edge tensors, returning shape (...,).

        displacement is R_j-R_i (...,3), A and B are (...,), coefficients
        (...,8), frames (...,3,3) with orthonormal columns. B must be positive
        and separation nonzero. Frame construction and domain validation belong
        at the input boundary, not inside the compiled energy kernel. The j end
        sees the NEGATIVE displacement, including the odd-l sign change.
        """
        local_i = torch.einsum('...i,...ij->...j', displacement, frame_i)
        local_j = torch.einsum('...i,...ij->...j', -displacement, frame_j)
        angular_i = 1 + (self.harmonics(local_i) * coefficients_i * self.mask_i).sum(-1)
        angular_j = 1 + (self.harmonics(local_j) * coefficients_j * self.mask_j).sum(-1)
        r = torch.linalg.vector_norm(displacement, dim=-1)
        x = torch.sqrt(b_i * b_j) * r
        overlap = (1 + x + x.square() / 3) * torch.exp(-x)
        return a_i * a_j * angular_i * angular_j * overlap
