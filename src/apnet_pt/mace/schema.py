"""Typed contracts shared by the MACE/AP3D3 integration.

This module deliberately has no dependency on :mod:`mace`.  Importing the base
``apnet_pt`` package therefore remains possible without the optional MACE stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

import torch


COMPONENT_ORDER = ("elst", "exch", "indu", "disp")
QUADRUPOLE_CONVENTION = "cartesian-symmetric-traceless-3x3"
SCF_CONVERGENCE_NORMS = ("l2", "rms", "max")
DEFAULT_SCF_CONVERGENCE_NORM = "l2"
INDUCTION_MODELS = ("ap3-no-correction", "cliff2-rackers")
DEFAULT_INDUCTION_MODEL = "ap3-no-correction"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PhysicsConfig:
    """Immutable definition of the classical and residual energy contract."""

    electrostatics_mode: Literal[
        "damped-cliff", "damped-amoeba", "undamped"
    ] = "damped-cliff"
    electrostatics_parameters: tuple[float, ...] = ()
    full_pair_edge_semantics: str = "all-intermonomer-pairs"
    polarizability_rule: str = "hfvr-4/3"
    thole_direct: float = 0.34
    thole_mutual: float = 0.39
    scf_tolerance: float = 1.0e-8
    scf_max_iterations: int = 200
    scf_nonconvergence: Literal["raise", "warn"] = "raise"
    scf_convergence_norm: Literal["l2", "rms", "max"] = DEFAULT_SCF_CONVERGENCE_NORM
    induction_model: Literal[
        "ap3-no-correction", "cliff2-rackers"
    ] = DEFAULT_INDUCTION_MODEL
    d3_parameters: tuple[float, ...] = ()
    neural_cutoff: float = 8.0
    component_order: tuple[str, ...] = COMPONENT_ORDER
    length_unit: str = "angstrom"
    energy_unit: str = "kcal/mol"
    quadrupole_convention: str = QUADRUPOLE_CONVENTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "electrostatics_parameters", tuple(self.electrostatics_parameters)
        )
        try:
            d3_parameters = tuple(float(value) for value in self.d3_parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError("d3_parameters must be numeric") from exc
        object.__setattr__(self, "d3_parameters", d3_parameters)
        object.__setattr__(self, "component_order", tuple(self.component_order))
        if self.electrostatics_mode not in {
            "damped-cliff",
            "damped-amoeba",
            "undamped",
        }:
            raise ValueError(
                f"Unsupported electrostatics_mode: {self.electrostatics_mode}"
            )
        if self.electrostatics_parameters:
            raise ValueError(
                "electrostatics_parameters are not active for the supported kernels"
            )
        if self.full_pair_edge_semantics != "all-intermonomer-pairs":
            raise ValueError(
                "full_pair_edge_semantics must be 'all-intermonomer-pairs'"
            )
        if self.polarizability_rule != "hfvr-4/3":
            raise ValueError("polarizability_rule must be 'hfvr-4/3'")
        if self.component_order != COMPONENT_ORDER:
            raise ValueError(
                f"component_order must be {COMPONENT_ORDER}, got {self.component_order}"
            )
        if self.length_unit != "angstrom" or self.energy_unit != "kcal/mol":
            raise ValueError("MACE/AP3D3 uses angstrom and kcal/mol at public seams")
        if self.quadrupole_convention != QUADRUPOLE_CONVENTION:
            raise ValueError(
                "quadrupole_convention must be Cartesian symmetric-traceless 3x3"
            )
        finite_positive = {
            "thole_direct": self.thole_direct,
            "thole_mutual": self.thole_mutual,
            "scf_tolerance": self.scf_tolerance,
            "neural_cutoff": self.neural_cutoff,
        }
        for name, value in finite_positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.scf_max_iterations, bool)
            or not isinstance(self.scf_max_iterations, int)
            or self.scf_max_iterations < 1
        ):
            raise ValueError("scf_max_iterations must be a positive integer")
        if self.scf_nonconvergence not in {"raise", "warn"}:
            raise ValueError("scf_nonconvergence must be 'raise' or 'warn'")
        if self.scf_convergence_norm not in SCF_CONVERGENCE_NORMS:
            raise ValueError(
                "scf_convergence_norm must be one of "
                f"{list(SCF_CONVERGENCE_NORMS)}, got {self.scf_convergence_norm!r}"
            )
        if self.induction_model not in INDUCTION_MODELS:
            raise ValueError(
                "induction_model must be one of "
                f"{list(INDUCTION_MODELS)}, got {self.induction_model!r}"
            )
        if len(self.d3_parameters) not in {0, 4}:
            raise ValueError("d3_parameters must be empty or (s6, s8, a1, a2)")
        if not all(math.isfinite(float(value)) for value in self.d3_parameters):
            raise ValueError("d3_parameters must be finite")

    @property
    def physics_hash(self) -> str:
        """Return a deterministic SHA-256 of every physics field.

        ``scf_convergence_norm`` and ``induction_model`` are elided at their
        defaults.  Each default reproduces exactly what the route did before
        the control existed -- the ``"l2"`` branch of the solver is the op
        sequence the loop used, and ``"ap3-no-correction"`` is the kernel
        ``_default_backends`` has always bound -- so a default-configured
        config is the same physics as one written before these fields, and
        elision keeps the hashes stamped into existing manifests and v3
        checkpoints valid.  Any non-default value is a different solver or a
        different induction functional, so it changes the hash.
        """

        fields = asdict(self)
        for name, default in (
            ("scf_convergence_norm", DEFAULT_SCF_CONVERGENCE_NORM),
            ("induction_model", DEFAULT_INDUCTION_MODEL),
        ):
            if fields[name] == default:
                del fields[name]
        return _canonical_hash(fields)


@dataclass(frozen=True)
class MACEAtomicFeatures:
    """Versioned isolated-monomer features returned by a MACE adapter."""

    invariant: torch.Tensor
    equivariant: torch.Tensor
    batch: torch.Tensor
    atomic_numbers: torch.Tensor
    total_charge: torch.Tensor
    total_spin: torch.Tensor
    feature_schema: str

    def __post_init__(self) -> None:
        if self.invariant.ndim != 2:
            raise ValueError("invariant features must have shape [n_atom, d]")
        if self.equivariant.ndim < 2:
            raise ValueError("equivariant features must have rank >= 2")
        if self.batch.ndim != 1 or self.batch.dtype not in {torch.int32, torch.int64}:
            raise ValueError("batch must be a rank-1 integer tensor")
        if (
            self.atomic_numbers.ndim != 1
            or self.atomic_numbers.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("atomic_numbers must be a rank-1 integer tensor")
        if self.total_charge.ndim != 1 or self.total_spin.ndim != 1:
            raise ValueError("total_charge and total_spin must be rank-1 tensors")
        float_tensors = {
            "invariant": self.invariant,
            "equivariant": self.equivariant,
            "total_charge": self.total_charge,
            "total_spin": self.total_spin,
        }
        for name, tensor in float_tensors.items():
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must use a floating dtype")
            if tensor.device != self.invariant.device:
                raise ValueError("all feature tensors must be on the same device")
            if tensor.dtype != self.invariant.dtype:
                raise ValueError("all floating feature tensors must use the same dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        for tensor in (self.batch, self.atomic_numbers):
            if tensor.device != self.invariant.device:
                raise ValueError("all feature tensors must be on the same device")
        natom = self.invariant.shape[0]
        for name, tensor in {
            "equivariant": self.equivariant,
            "batch": self.batch,
            "atomic_numbers": self.atomic_numbers,
        }.items():
            if tensor.shape[0] != natom:
                raise ValueError(f"{name} first dimension must equal n_atom={natom}")
        if natom and (self.batch.min() < 0 or self.atomic_numbers.min() <= 0):
            raise ValueError("batch indices and atomic numbers must be non-negative/positive")
        if not self.feature_schema:
            raise ValueError("feature_schema must be non-empty")
        nmonomer = int(self.batch.max().item()) + 1 if natom else 0
        if self.total_charge.numel() != nmonomer or self.total_spin.numel() != nmonomer:
            raise ValueError("charge/spin must contain one value per isolated monomer")

    @property
    def natom(self) -> int:
        return self.invariant.shape[0]


POLAR_DENSITY_L1_CONTRACT = "polar-density-l1-yzx-eangstrom-v1"


@dataclass(frozen=True)
class PolarMACEDirectOutputs:
    """Validated public PolarMACE charge and dipole outputs.

    The first density coefficient is charge and coefficients ``[3, 1, 2]``
    are intrinsic Cartesian ``x, y, z`` dipoles in eÅ.
    """

    density_coefficients: torch.Tensor
    charges: torch.Tensor
    molecular_dipole_eangstrom: torch.Tensor
    positions_angstrom: torch.Tensor
    batch: torch.Tensor
    total_charge: torch.Tensor
    multipole_contract: str = POLAR_DENSITY_L1_CONTRACT

    def __post_init__(self) -> None:
        if self.multipole_contract != POLAR_DENSITY_L1_CONTRACT:
            raise ValueError(
                f"incompatible PolarMACE multipole contract: {self.multipole_contract}"
            )
        natom = self.charges.numel()
        expected = {
            "density_coefficients": (natom, 4),
            "charges": (natom,),
            "positions_angstrom": (natom, 3),
            "batch": (natom,),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if (
            self.molecular_dipole_eangstrom.ndim != 2
            or self.molecular_dipole_eangstrom.shape[1] != 3
        ):
            raise ValueError("molecular dipole must have shape [n_monomer, 3]")
        nmonomer = int(self.batch.max().item()) + 1 if natom else 0
        if self.molecular_dipole_eangstrom.shape[0] != nmonomer:
            raise ValueError("molecular dipole must contain one value per monomer")
        if self.total_charge.shape != (nmonomer,):
            raise ValueError("total_charge must contain one value per monomer")
        if natom and self.batch.min() < 0:
            raise ValueError("direct-output batch indices must be non-negative")
        if natom and self.batch.unique(sorted=True).tolist() != list(range(nmonomer)):
            raise ValueError("direct-output batch indices must be contiguous")
        if self.batch.dtype not in {torch.int32, torch.int64}:
            raise ValueError("direct-output batch must be an integer tensor")
        for name in (
            "density_coefficients",
            "charges",
            "molecular_dipole_eangstrom",
            "positions_angstrom",
            "total_charge",
        ):
            tensor = getattr(self, name)
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must use a floating dtype")
            if tensor.dtype != self.density_coefficients.dtype:
                raise ValueError("direct PolarMACE outputs must share a floating dtype")
            if tensor.device != self.density_coefficients.device:
                raise ValueError("direct PolarMACE outputs must share a device")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        if self.batch.device != self.density_coefficients.device:
            raise ValueError("direct-output batch must share the output device")
        if not torch.allclose(
            self.density_coefficients[:, 0], self.charges, atol=1.0e-7, rtol=1.0e-6
        ):
            raise ValueError("density coefficient zero must equal public charges")

    @property
    def intrinsic_dipole_eangstrom(self) -> torch.Tensor:
        """Return intrinsic atomic dipoles in Cartesian x/y/z ordering."""

        return self.density_coefficients[:, [3, 1, 2]]


@dataclass(frozen=True)
class AtomicPropertyBundle:
    """QCMLForge atom properties in the canonical AP3 Cartesian convention."""

    q: torch.Tensor
    mu: torch.Tensor
    quadrupole: torch.Tensor
    hfvr: torch.Tensor
    valence_width: torch.Tensor
    alpha: torch.Tensor
    damping: torch.Tensor

    def __post_init__(self) -> None:
        if self.q.ndim != 2:
            raise ValueError("q must have rank 2 and shape [n_atom, 1]")
        natom = self.q.shape[0]
        expected = {
            "q": (natom, 1),
            "mu": (natom, 3),
            "quadrupole": (natom, 3, 3),
            "hfvr": (natom, 1),
            "valence_width": (natom, 1),
            "alpha": (natom, 1),
            "damping": (natom, 1),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
            if tensor.device != self.q.device:
                raise ValueError("all atomic properties must be on the same device")
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must use a floating dtype")
            if tensor.dtype != self.q.dtype:
                raise ValueError("all atomic properties must use the same dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")

    @property
    def natom(self) -> int:
        return self.q.shape[0]

    def cloned(self) -> "AtomicPropertyBundle":
        """Deep-clone tensors before calling legacy kernels that may mutate inputs."""

        return AtomicPropertyBundle(
            **{name: getattr(self, name).clone() for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class InductionDiagnostics:
    converged: bool
    iterations: int
    residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be bool")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise ValueError("iterations must be an integer")
        if self.iterations < 0 or not math.isfinite(self.residual) or self.residual < 0:
            raise ValueError("induction diagnostics must be finite and non-negative")


@dataclass(frozen=True)
class ClassicalEnergyBundle:
    """Single-source pair and dimer ledgers for classical SAPT terms."""

    pair_elst: torch.Tensor
    pair_ind: torch.Tensor
    pair_disp: torch.Tensor
    dimer_elst: torch.Tensor
    dimer_ind: torch.Tensor
    dimer_disp: torch.Tensor
    induction_diagnostics: InductionDiagnostics
    physics_config_hash: str = ""

    def __post_init__(self) -> None:
        tensors = {
            "pair_elst": self.pair_elst,
            "pair_ind": self.pair_ind,
            "pair_disp": self.pair_disp,
            "dimer_elst": self.dimer_elst,
            "dimer_ind": self.dimer_ind,
            "dimer_disp": self.dimer_disp,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 1 or not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must be a rank-1 floating tensor")
            if tensor.device != self.pair_elst.device or tensor.dtype != self.pair_elst.dtype:
                raise ValueError("all classical ledgers must share dtype and device")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        npair = self.pair_elst.numel()
        if self.pair_ind.numel() != npair or self.pair_disp.numel() != npair:
            raise ValueError("classical pair ledgers must use the same full edge list")
        if self.physics_config_hash and (
            len(self.physics_config_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.physics_config_hash)
        ):
            raise ValueError("physics_config_hash must be a lowercase SHA-256 digest")
        ndimer = self.dimer_elst.numel()
        if self.dimer_ind.numel() != ndimer or self.dimer_disp.numel() != ndimer:
            raise ValueError("classical dimer ledgers must have matching lengths")


@dataclass(frozen=True)
class MACEFeatureCacheKey:
    """Exact, isolated-monomer cache identity.

    Coordinates are intentionally not quantized.  Rotations, translations, atom
    permutations, charge, spin, dtype, or schema changes are cache misses.
    """

    checkpoint_sha256: str
    mace_version: str
    feature_schema: str
    physics_config_hash: str
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    total_charge: float
    total_spin: float
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "atomic_numbers", tuple(self.atomic_numbers))
        object.__setattr__(
            self,
            "coordinates_angstrom",
            tuple(tuple(coordinate) for coordinate in self.coordinates_angstrom),
        )
        for name in ("checkpoint_sha256", "physics_config_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if len(self.atomic_numbers) != len(self.coordinates_angstrom):
            raise ValueError("cache key atom numbers and coordinates must align")
        if any(number <= 0 for number in self.atomic_numbers):
            raise ValueError("cache key atomic numbers must be positive")
        if not math.isfinite(self.total_charge) or not math.isfinite(self.total_spin):
            raise ValueError("cache key charge and spin must be finite")
        if any(
            not math.isfinite(value)
            for coordinate in self.coordinates_angstrom
            for value in coordinate
        ):
            raise ValueError("cache key coordinates must be finite")
        if self.dtype not in {"torch.float32", "torch.float64"}:
            raise ValueError("cache key dtype must be torch.float32 or torch.float64")

    @classmethod
    def from_tensors(
        cls,
        *,
        checkpoint_sha256: str,
        mace_version: str,
        feature_schema: str,
        physics_config_hash: str,
        atomic_numbers: torch.Tensor,
        coordinates_angstrom: torch.Tensor,
        total_charge: float,
        total_spin: float,
        dtype: torch.dtype,
    ) -> "MACEFeatureCacheKey":
        if atomic_numbers.ndim != 1 or atomic_numbers.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("atomic_numbers must be a rank-1 integer tensor")
        if coordinates_angstrom.shape != (atomic_numbers.numel(), 3):
            raise ValueError("coordinates must have shape [n_atom, 3]")
        if not torch.is_floating_point(coordinates_angstrom):
            raise ValueError("coordinates must use a floating dtype")
        if atomic_numbers.device != coordinates_angstrom.device:
            raise ValueError("cache key tensors must use the same device")
        return cls(
            checkpoint_sha256=checkpoint_sha256,
            mace_version=mace_version,
            feature_schema=feature_schema,
            physics_config_hash=physics_config_hash,
            atomic_numbers=tuple(int(v) for v in atomic_numbers.detach().cpu().tolist()),
            coordinates_angstrom=tuple(
                tuple(float(v) for v in row)
                for row in coordinates_angstrom.detach().cpu().tolist()
            ),
            total_charge=float(total_charge),
            total_spin=float(total_spin),
            dtype=str(dtype),
        )

    @property
    def cache_hash(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class ExternalMACEArtifact:
    path: str
    model_id: str
    sha256: str
    mace_version: str
    license: str = "ASL"
    source_url: str = ""
    acknowledged: bool = False
