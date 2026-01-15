import qcelemental 
import torch
import os
h2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstrom = qcelemental.constants.bohr2angstroms

from .data import radii, r4r2
from .rational import rational_damping
from .weights import weight_references 
from . import defaults


param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
}

def get_distances(RA, RB, e_source, e_target):
        """
        Compute pairwise displacement vectors and Euclidean distances between selected rows of two coordinate tensors.
        
        Parameters:
            RA (torch.Tensor): Source coordinates with shape (N_source_total, 3).
            RB (torch.Tensor): Target coordinates with shape (N_target_total, 3).
            e_source (torch.Tensor): 1-D index tensor selecting rows from `RA`.
            e_target (torch.Tensor): 1-D index tensor selecting rows from `RB`; must be the same length as `e_source`.
        
        Returns:
            tuple:
                dR (torch.Tensor): 1-D tensor of Euclidean distances for each selected pair (len == len(e_source)). Distances are computed from the displacement vectors and clamped for numerical stability.
                dR_xyz (torch.Tensor): Tensor of displacement vectors RB[e_target] - RA[e_source] with shape (len(e_source), 3).
        """
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz


def exp_count(
    distances: torch.tensor, 
    cov_r: torch.tensor, 
) -> torch.tensor:
    
    """
    Compute a smooth, distance-dependent neighbor contribution used for coordination numbers.
    
    Calculates a sigmoidal weight in (0, 1) that quantifies how strongly a pair of atoms contributes to a coordination number based on their interatomic distance and a reference covalent radius sum.
    
    Parameters:
        distances (torch.tensor): Pairwise distances (same shape as cov_r) in the same length scale.
        cov_r (torch.tensor): Reference covalent distance (typically sum of covalent radii) for each pair.
    
    Returns:
        torch.tensor: A tensor of the same shape as the inputs with values in (0, 1); larger values indicate stronger neighbor contribution.
    """
    k2 = 4.0 / 3.0 #ad hoc factor so the cn is reasonable for molecules
    k1 = 16 #large so distant atoms are not counted so CN does not depend on size of system
    
    return 1.0 / (1.0 + torch.exp(-k1 * (torch.divide(k2 * cov_r, distances) - 1.0)))

def cn_d3_intermolecular(
    batch,
) -> torch.tensor:
    
    """
    Compute intermolecular D3 coordination numbers (CN) for atoms in a batch.
    
    Calculates smooth, distance-weighted coordination numbers for atom pairs across two fragments (A and B) using covalent D3 radii and an exponential counting function; contributions beyond the D3 CN cutoff are ignored, and per-pair contributions are summed into per-atom CN arrays for both fragments.
    
    Parameters:
        batch: Batch-like object providing RA, RB (atom coordinates), ZA, ZB (atomic numbers),
            and index tensors e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target
            that define the source/target pairs for short-range and long-range interactions.
    
    Returns:
        tuple: (cn_A, cn_B)
            cn_A (torch.Tensor): 1D tensor of per-atom coordination numbers for fragment A.
            cn_B (torch.Tensor): 1D tensor of per-atom coordination numbers for fragment B.
    """
    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

    cutoff = torch.tensor(defaults.D3_CN_CUTOFF, **dd)


    e_source_full = torch.concatenate([batch.e_ABsr_source, batch.e_ABlr_source,])
    e_target_full = torch.concatenate([batch.e_ABsr_target, batch.e_ABlr_target,])
    
    ZA = batch.ZA
    ZB = batch.ZB
    RA = batch.RA
    RB = batch.RB

    ZA = ZA.index_select(0, e_source_full)
    ZB = ZB.index_select(0, e_target_full)
    RA = RA.index_select(0, e_source_full)
    RB = RB.index_select(0, e_target_full)

    rcov = radii.COV_D3(**dd)[ZA] + radii.COV_D3(**dd)[ZB] 
    
    distances, _ = get_distances(RA, RB, e_source_full, e_target_full)
    cn = torch.where(
        (distances <= cutoff),
        exp_count(distances, rcov),
        torch.tensor(0.0, **dd)
    )

    size = e_source_full.max().item() + 1
    cn_A = torch.zeros(size, dtype=cn.dtype)
    cn_A.scatter_reduce_(0, e_source_full, cn, reduce="sum", include_self=False)
    
    cn_B = torch.zeros(size, dtype=cn.dtype)
    cn_B.scatter_reduce_(0, e_target_full, cn, reduce="sum", include_self=False)
    return cn_A, cn_B


def d3(
    batch,
):
    """
    Compute D3 pairwise dispersion energy contributions for intermolecular atom pairs in a batch.
    
    Calculates intermolecular C6 and C8 coefficients using reference C6 data and coordination-number-dependent atomic weights, applies rational damping for orders 6 and 8, combines the damped terms with global scaling factors, and converts energies to kcal/mol.
    
    Parameters:
        batch: A data container providing per-atom positions, atomic numbers, and index lists. Expected attributes include:
            - RA, RB: atomic coordinates (in Bohr) for fragments A and B
            - ZA, ZB: atomic numbers for fragments A and B
            - e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target: index tensors defining short- and long-range interacting pairs
            - other fields consumed by cn_d3_intermolecular
    
    Returns:
        torch.Tensor: 1-D tensor of pairwise D3 dispersion energy contributions (kcal/mol) for the concatenated source/target interaction list.
    """
    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

    path = os.path.join(os.path.dirname(__file__), "data/reference-c6.pt")
    kwargs = {"weights_only" : True, "map_location" : dd['device']}
    ref_c6 = torch.load(path, **kwargs).type(dtype=dd['dtype'])

    cn_A, cn_B = cn_d3_intermolecular(
        batch,
    ) 

    ZA = batch.ZA
    RA = batch.RA / bohr2angstrom

    ZB = batch.ZB
    RB = batch.RB / bohr2angstrom
    
    e_source_full = torch.concatenate([batch.e_ABsr_source, batch.e_ABlr_source,])
    e_target_full = torch.concatenate([batch.e_ABsr_target, batch.e_ABlr_target,])
    cn_A = cn_A.index_select(0, e_source_full)

    cn_B = cn_B.index_select(0, e_target_full)    
    ZA = ZA.index_select(0, e_source_full)
    ZB = ZB.index_select(0, e_target_full)

    
    weights_A = weight_references(ZA, cn_A,)
    weights_B = weight_references(ZB, cn_B,)

    rc6 = ref_c6[ZA, ZB]
    c6 = torch.einsum("ijk,ij,ik->i", rc6, weights_A, weights_B)
    distances, _ = get_distances(RA=RA, RB=RB, e_source=e_source_full, e_target=e_target_full)

    #C8 is computed recursively from c6

    #Q_A = sqrt(Z) * r^4/r^2
    r4_over_r2 = r4r2.R4R2(**dd)
    #ad hoc nuclear charge dependent factor
    sqrtz = torch.sqrt(
        torch.arange(len(r4_over_r2), **dd)
    )
    Q = r4_over_r2 * sqrtz
    #C_8 = 3 * C_6 * sqrt(Q_A * Q_B)

    #quotient of C8 and C6, used later by damping function
    qAqB = 3 * torch.sqrt((Q[ZA] * Q[ZB]))
    c8 = c6 * qAqB
   
    t6 = rational_damping(6, distances, qAqB, param,)
    t8 = rational_damping(8, distances, qAqB, param,)
    
    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
    e6 = -1 * (c6 * t6) * s6
    e8 = -1 * (c8 * t8) * s8
    pairwise_energies = e6 + e8
    pairwise_energies *= h2kcalmol
    return pairwise_energies