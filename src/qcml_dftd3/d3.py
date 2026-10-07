from functools import lru_cache
from importlib import resources

from .weights import weight_references
from . import defaults
from .rational import rational_damping
from .data import radii, r4r2
import qcelemental
import torch

h2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstrom = qcelemental.constants.bohr2angstroms


params_intermolecular_saptpbe0_d3i = {
    "s6": 1.0,
    "s8": 0.8614,
    "a1": 0.7171,
    "a2": 0.5375,
}

params_intermolecular_sapt0_d3i = {
    "s6": 1.0,
    "s8": 0.9428623751222317,
    "a1": 0.33993637135556765,
    "a2": 3.0374641668809055,
}


def _to_python_float(value) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("D3 damping parameter tensors must be scalar-valued")
        return float(value.detach().cpu().item())
    return float(value)


D3_DAMPING_PARAMETER_SETS = {
    "sapt-pbe0-d3i": params_intermolecular_saptpbe0_d3i,
}
"""Named intermolecular D3(BJ) damping parameter sets."""


def resolve_d3_damping_parameters(
    params: dict | str | None = None,
) -> dict[str, float]:
    """
    Resolve D3(BJ) damping parameters, defaulting to ``"sapt-pbe0-d3i"``.

    Parameters
    ----------
    params : dict, str or None
        ``None`` for the default set, a key of ``D3_DAMPING_PARAMETER_SETS``,
        or a mapping overriding any of ``s6``, ``s8``, ``a1`` and ``a2``.

    Returns
    -------
    dict[str, float]
        A new dictionary holding all four parameters.
    """
    if isinstance(params, str):
        if params not in D3_DAMPING_PARAMETER_SETS:
            raise ValueError(
                f"Unknown D3 damping parameter set {params!r}; "
                f"choose from {sorted(D3_DAMPING_PARAMETER_SETS)}"
            )
        return dict(D3_DAMPING_PARAMETER_SETS[params])
    resolved = dict(params_intermolecular_saptpbe0_d3i)
    if params is None:
        return resolved

    allowed_keys = set(resolved)
    for key, value in params.items():
        if key not in allowed_keys:
            raise ValueError(f"Unknown D3 damping parameter: {key}")
        resolved[key] = _to_python_float(value)
    return resolved


@lru_cache(maxsize=1)
def _load_reference_c6_cpu() -> torch.Tensor:
    ref_path = resources.files("qcml_dftd3.data").joinpath("reference-c6.pt")
    with ref_path.open("rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=True)


_REF_C6_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _get_reference_c6(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cache_key = (str(device), str(dtype))
    ref_c6 = _REF_C6_CACHE.get(cache_key)
    if ref_c6 is None:
        ref_c6 = _load_reference_c6_cpu().to(device=device, dtype=dtype)
        _REF_C6_CACHE[cache_key] = ref_c6
    return ref_c6


def get_distances(RA, RB, e_source, e_target):
    RA_source = RA.index_select(0, e_source)
    RB_target = RB.index_select(0, e_target)
    dR_xyz = RB_target - RA_source

    # Compute distances with safe operation for square root
    # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
    dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
    return dR, dR_xyz


def exp_count(
    distances: torch.Tensor,
    cov_r: torch.Tensor,
) -> torch.Tensor:

    k2 = 4.0 / 3.0  # ad hoc factor so the cn is reasonable for molecules
    k1 = 16  # large so distant atoms are not counted so CN does not depend on size of system

    return 1.0 / (1.0 + torch.exp(-k1 * (torch.divide(k2 * cov_r, distances) - 1.0)))


def cn_d3_intermolecular(
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:

    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

    cutoff = torch.tensor(defaults.D3_CN_CUTOFF, **dd)

    # Intermolecular edges (A->B)
    if hasattr(batch, "e_ABfull_source"):
        e_AB_source = batch.e_ABfull_source
        e_AB_target = batch.e_ABfull_target
    else:
        e_AB_source = torch.concatenate(
            [
                batch.e_ABsr_source,
                batch.e_ABlr_source,
            ]
        )
        e_AB_target = torch.concatenate(
            [
                batch.e_ABsr_target,
                batch.e_ABlr_target,
            ]
        )

    ZA = batch.ZA
    ZB = batch.ZB
    # Convert coordinates from angstrom to bohr (covalent radii and cutoff are in bohr)
    RA = batch.RA / bohr2angstrom
    RB = batch.RB / bohr2angstrom

    rcov = radii.COV_D3(**dd)

    # --- CN for monomer A atoms ---
    # Contribution from intramolecular A-A edges
    cn_A = torch.zeros(len(batch.ZA), **dd)

    if hasattr(batch, "e_AA_source") and len(batch.e_AA_source) > 0:
        e_AA_s = batch.e_AA_source
        e_AA_t = batch.e_AA_target
        ZA_AA_s = ZA.index_select(0, e_AA_s)
        ZA_AA_t = ZA.index_select(0, e_AA_t)
        RA_AA_s = RA.index_select(0, e_AA_s)
        RA_AA_t = RA.index_select(0, e_AA_t)
        rcov_AA = rcov[ZA_AA_s] + rcov[ZA_AA_t]
        dR_AA = torch.sqrt(torch.sum((RA_AA_t - RA_AA_s) ** 2, dim=-1).clamp_min(1e-10))
        cn_AA = torch.where(
            (dR_AA <= cutoff), exp_count(dR_AA, rcov_AA), torch.tensor(0.0, **dd)
        )
        cn_A.scatter_add_(0, e_AA_s, cn_AA)

    # Contribution from intermolecular A->B edges (A atom is source)
    if len(e_AB_source) > 0:
        ZA_AB = ZA.index_select(0, e_AB_source)
        ZB_AB = ZB.index_select(0, e_AB_target)
        RA_AB = RA.index_select(0, e_AB_source)
        RB_AB = RB.index_select(0, e_AB_target)
        rcov_AB = rcov[ZA_AB] + rcov[ZB_AB]
        dR_AB = torch.sqrt(torch.sum((RB_AB - RA_AB) ** 2, dim=-1).clamp_min(1e-10))
        cn_AB_vals = torch.where(
            (dR_AB <= cutoff), exp_count(dR_AB, rcov_AB), torch.tensor(0.0, **dd)
        )
        cn_A.scatter_add_(0, e_AB_source, cn_AB_vals)

    # --- CN for monomer B atoms ---
    # Contribution from intramolecular B-B edges
    cn_B = torch.zeros(len(batch.ZB), **dd)

    if hasattr(batch, "e_BB_source") and len(batch.e_BB_source) > 0:
        e_BB_s = batch.e_BB_source
        e_BB_t = batch.e_BB_target
        ZB_BB_s = ZB.index_select(0, e_BB_s)
        ZB_BB_t = ZB.index_select(0, e_BB_t)
        RB_BB_s = RB.index_select(0, e_BB_s)
        RB_BB_t = RB.index_select(0, e_BB_t)
        rcov_BB = rcov[ZB_BB_s] + rcov[ZB_BB_t]
        dR_BB = torch.sqrt(torch.sum((RB_BB_t - RB_BB_s) ** 2, dim=-1).clamp_min(1e-10))
        cn_BB = torch.where(
            (dR_BB <= cutoff), exp_count(dR_BB, rcov_BB), torch.tensor(0.0, **dd)
        )
        cn_B.scatter_add_(0, e_BB_s, cn_BB)

    # Contribution from intermolecular A->B edges (B atom is target)
    if len(e_AB_target) > 0:
        ZA_BA = ZA.index_select(0, e_AB_source)
        ZB_BA = ZB.index_select(0, e_AB_target)
        RA_BA = RA.index_select(0, e_AB_source)
        RB_BA = RB.index_select(0, e_AB_target)
        rcov_BA = rcov[ZA_BA] + rcov[ZB_BA]
        dR_BA = torch.sqrt(torch.sum((RB_BA - RA_BA) ** 2, dim=-1).clamp_min(1e-10))
        cn_BA_vals = torch.where(
            (dR_BA <= cutoff), exp_count(dR_BA, rcov_BA), torch.tensor(0.0, **dd)
        )
        cn_B.scatter_add_(0, e_AB_target, cn_BA_vals)

    return cn_A, cn_B


def d3_pair_terms(batch) -> dict[str, torch.Tensor]:
    """
    Damping-independent intermolecular D3 pair terms of a dimer batch.

    Returns
    -------
    dict[str, torch.Tensor]
        Per A->B pair: ``c6`` (hartree bohr^6), ``qq`` (C8/C6 quotient,
        bohr^2) and ``distances`` (bohr).
    """
    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

    ref_c6 = _get_reference_c6(dd["device"], dd["dtype"])

    cn_A, cn_B = cn_d3_intermolecular(
        batch,
    )

    ZA = batch.ZA
    RA = batch.RA / bohr2angstrom

    ZB = batch.ZB
    RB = batch.RB / bohr2angstrom

    if hasattr(batch, "e_ABfull_source"):
        e_source_full = batch.e_ABfull_source
        e_target_full = batch.e_ABfull_target
    else:
        e_source_full = torch.concatenate(
            [
                batch.e_ABsr_source,
                batch.e_ABlr_source,
            ]
        )
        e_target_full = torch.concatenate(
            [
                batch.e_ABsr_target,
                batch.e_ABlr_target,
            ]
        )
    cn_A = cn_A.index_select(0, e_source_full)

    cn_B = cn_B.index_select(0, e_target_full)
    ZA = ZA.index_select(0, e_source_full)
    ZB = ZB.index_select(0, e_target_full)

    weights_A = weight_references(
        ZA,
        cn_A,
    )
    weights_B = weight_references(
        ZB,
        cn_B,
    )

    rc6 = ref_c6[ZA, ZB]
    c6 = torch.einsum("ijk,ij,ik->i", rc6, weights_A, weights_B)
    distances, _ = get_distances(
        RA=RA, RB=RB, e_source=e_source_full, e_target=e_target_full
    )

    # C8 is computed recursively from c6

    # Fortran: rrij = 3*r4r2(izp)*r4r2(jzp)
    # R4R2() already returns sqrt(0.5 * raw * sqrtZ), matching Fortran r4r2 values
    r4_over_r2 = r4r2.R4R2(**dd)

    # quotient of C8 and C6: qAqB = 3 * r4r2[A] * r4r2[B]
    qAqB = 3 * r4_over_r2[ZA] * r4_over_r2[ZB]
    return {"c6": c6, "qq": qAqB, "distances": distances}


def d3_pair_energies(terms: dict[str, torch.Tensor], params: dict) -> torch.Tensor:
    """
    Two-body D3(BJ) energies (kcal/mol) from ``d3_pair_terms`` output.

    ``params`` must hold ``s6``, ``s8``, ``a1`` and ``a2``; tensor values keep
    their autograd graph, so damping parameters can be fitted directly.
    """
    c6 = terms["c6"]
    qAqB = terms["qq"]
    distances = terms["distances"]
    c8 = c6 * qAqB

    t6 = rational_damping(
        6,
        distances,
        qAqB,
        params,
    )
    t8 = rational_damping(
        8,
        distances,
        qAqB,
        params,
    )

    e6 = -1 * (c6 * t6) * params["s6"]
    e8 = -1 * (c8 * t8) * params["s8"]
    pairwise_energies = e6 + e8
    pairwise_energies *= h2kcalmol
    return pairwise_energies


def d3(
    batch,
    params=params_intermolecular_saptpbe0_d3i,
):
    """
    Pairwise intermolecular D3(BJ) dispersion energies (kcal/mol).

    ``params`` is anything ``resolve_d3_damping_parameters`` accepts, e.g. a
    mapping or the name of an entry in ``D3_DAMPING_PARAMETER_SETS``.
    """
    params = resolve_d3_damping_parameters(params)
    return d3_pair_energies(d3_pair_terms(batch), params)
