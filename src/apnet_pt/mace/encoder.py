"""Verified, optional PolarMACE loading helpers.

MACE imports occur only after a local artifact has passed digest verification.
The foundation checkpoint is external and must never be embedded in QCMLForge
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, MutableMapping

import torch

from apnet_pt.constants import ALLOWED_ELEMENTS

from .schema import (
    MACEAtomicFeatures,
    MACEFeatureCacheKey,
    POLAR_DENSITY_L1_CONTRACT,
    PhysicsConfig,
    PolarMACEDirectOutputs,
)


POLAR_1S_MODEL_ID = "polar-1-s"
POLAR_1S_SHA256 = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
POLAR_1S_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_polar_1/MACE-POLAR-1-S.model"
)
SUPPORTED_MACE_VERSION = "0.3.16"


def _e3nn_o3():
    """Import pinned e3nn constants under PyTorch's restricted safe loader."""

    with torch.serialization.safe_globals([slice]):
        from e3nn import o3

    return o3


@dataclass(frozen=True)
class PrivateMACEFeatures:
    """Versioned result of the reviewed private-layer adapter."""

    final_scalars: torch.Tensor
    hidden: torch.Tensor
    hidden_irreps: str
    layer_count: int
    adapter_version: str


class PolarMACEPrivateLayerAdapter:
    """Reviewed MACE 0.3.16 local-interaction adapter.

    This adapter calls named blocks explicitly rather than installing forward
    hooks.  Its final product-basis scalars are checked against public
    ``node_feats`` on every use; a mismatch is fatal.
    """

    adapter_version = "polar-private-mace-0.3.16-v1"

    def __init__(self, mace_version: str) -> None:
        if mace_version != SUPPORTED_MACE_VERSION:
            raise ValueError(
                "private PolarMACE adapter supports only mace-torch 0.3.16"
            )
        self.version = self.adapter_version

    def extract(
        self,
        backbone: torch.nn.Module,
        graph: dict[str, torch.Tensor],
        public_outputs: dict[str, torch.Tensor],
    ) -> PrivateMACEFeatures:
        if (
            type(backbone).__name__ != "PolarMACE"
            or type(backbone).__module__ != "mace.modules.extensions"
        ):
            raise TypeError("private adapter requires the pinned PolarMACE class")
        if version("mace-torch") != SUPPORTED_MACE_VERSION:
            raise RuntimeError("private adapter MACE runtime version mismatch")
        # Optional/private imports are deliberately delayed until adapter use.
        o3 = _e3nn_o3()
        from mace.modules.extensions import _permute_to_e3nn_convention
        from mace.modules.utils import prepare_graph

        ctx = prepare_graph(
            graph,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        node_feats = backbone.node_embedding(graph["node_attrs"])
        edge_attrs = backbone.spherical_harmonics(
            _permute_to_e3nn_convention(ctx.vectors)
        )
        edge_feats, cutoff = backbone.radial_embedding(
            ctx.lengths,
            graph["node_attrs"],
            graph["edge_index"],
            backbone.atomic_numbers,
        )
        hidden_layers = []
        product_layers = []
        layer_irreps = []
        for index, (interaction, product) in enumerate(
            zip(backbone.interactions, backbone.products)
        ):
            hidden, sc = interaction(
                node_attrs=graph["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=graph["edge_index"],
                cutoff=cutoff,
                first_layer=(index == 0),
                lammps_class=ctx.interaction_kwargs.lammps_class,
                lammps_natoms=ctx.interaction_kwargs.lammps_natoms,
            )
            interaction_irreps = o3.Irreps(interaction.irreps_out)
            layer_irreps.append(interaction_irreps)
            if hidden.ndim == 3:
                # MACE 0.3.16's reshape block uses [atom, channel,
                # concatenated-m] layout. Convert it to the conventional e3nn
                # irrep-major flattened layout consumed by downstream heads.
                offset = 0
                flattened_parts = []
                for multiplicity, irrep in interaction_irreps:
                    if multiplicity != hidden.shape[1]:
                        raise RuntimeError(
                            "unsupported private PolarMACE channel layout"
                        )
                    flattened_parts.append(
                        hidden[:, :, offset : offset + irrep.dim].reshape(
                            hidden.shape[0], -1
                        )
                    )
                    offset += irrep.dim
                hidden_layers.append(torch.cat(flattened_parts, dim=-1))
            else:
                hidden_layers.append(hidden)
            node_feats = product(
                node_feats=hidden,
                sc=sc,
                node_attrs=graph["node_attrs"],
            )
            product_layers.append(node_feats)
        if not hidden_layers:
            raise ValueError("PolarMACE private adapter found no interaction layers")
        hidden_irreps = o3.Irreps("")
        for irreps in layer_irreps:
            hidden_irreps += irreps
        return PrivateMACEFeatures(
            final_scalars=torch.cat(product_layers, dim=-1),
            hidden=torch.cat(hidden_layers, dim=-1),
            hidden_irreps=str(hidden_irreps),
            layer_count=len(hidden_layers),
            adapter_version=self.adapter_version,
        )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact without deserializing or loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path, expected_sha256: str) -> str:
    """Verify a local artifact and return its normalized digest."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"MACE artifact does not exist: {artifact}")
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    actual = sha256_file(artifact)
    if actual != expected:
        raise ValueError(
            f"MACE artifact SHA-256 mismatch for {artifact}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _default_polar_loader(**kwargs):
    # Optional import intentionally occurs only after verify_artifact succeeds.
    from mace.calculators.foundations_models import mace_polar

    return mace_polar(**kwargs)


def load_verified_polar_mace(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
    offline: bool = True,
    loader: Callable[..., torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """Load a digest-verified local PolarMACE artifact as a frozen module.

    Parameters
    ----------
    path
        Local external checkpoint.  This function never downloads artifacts.
    expected_sha256
        Required trusted digest, checked before MACE is imported/deserializes.
    device
        Device passed to the upstream raw-module loader.
    offline
        Kept explicit in manifests/API.  Missing local files fail early in both
        modes; callers that permit downloads must resolve and verify separately.
    loader
        Test seam or version-pinned upstream loader.
    """

    artifact = Path(path)
    if not artifact.is_file():
        qualifier = " while offline" if offline else ""
        raise FileNotFoundError(
            f"Local MACE artifact is required{qualifier}: {artifact}"
        )
    verify_artifact(artifact, expected_sha256)

    load = loader or _default_polar_loader
    model = load(
        model=str(artifact),
        device=str(device),
        return_raw_model=True,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("PolarMACE loader did not return a torch.nn.Module")
    model.eval()
    model.requires_grad_(False)
    return model


def _clone_features(features: MACEAtomicFeatures) -> MACEAtomicFeatures:
    return MACEAtomicFeatures(
        invariant=features.invariant.clone(),
        equivariant=features.equivariant.clone(),
        batch=features.batch.clone(),
        atomic_numbers=features.atomic_numbers.clone(),
        total_charge=features.total_charge.clone(),
        total_spin=features.total_spin.clone(),
        feature_schema=features.feature_schema,
    )


def _clone_direct(outputs: PolarMACEDirectOutputs) -> PolarMACEDirectOutputs:
    return PolarMACEDirectOutputs(
        density_coefficients=outputs.density_coefficients.clone(),
        charges=outputs.charges.clone(),
        molecular_dipole_eangstrom=outputs.molecular_dipole_eangstrom.clone(),
        positions_angstrom=outputs.positions_angstrom.clone(),
        batch=outputs.batch.clone(),
        total_charge=outputs.total_charge.clone(),
        multipole_contract=outputs.multipole_contract,
    )


class MACEPolarFeaturizer(torch.nn.Module):
    """Frozen isolated-monomer PolarMACE feature and direct-output adapter."""

    valid_feature_modes = {"final-layer-scalars", "all-scalars+norms"}

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        checkpoint_sha256: str,
        mace_version: str = SUPPORTED_MACE_VERSION,
        model_id: str = POLAR_1S_MODEL_ID,
        feature_mode: str = "final-layer-scalars",
        dtype: torch.dtype = torch.float32,
        physics_config: PhysicsConfig | None = None,
        cache: MutableMapping[str, tuple[MACEAtomicFeatures, PolarMACEDirectOutputs]]
        | None = None,
        graph_builder: Callable[..., dict[str, torch.Tensor]] | None = None,
        private_adapter: Any | None = None,
        multipole_contract: str = POLAR_DENSITY_L1_CONTRACT,
        parity_atol: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if feature_mode not in self.valid_feature_modes:
            raise ValueError(f"unsupported MACE feature mode: {feature_mode}")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("MACE featurizer dtype must be float32 or float64")
        if multipole_contract != POLAR_DENSITY_L1_CONTRACT:
            raise ValueError("incompatible PolarMACE multipole contract")
        maximum_l = getattr(backbone, "atomic_multipoles_max_l", 1)
        if int(maximum_l) != 1:
            raise ValueError("PolarMACE artifact must provide direct multipoles through l=1")
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if graph_builder is None and type(backbone).__name__ != "PolarMACE":
            raise TypeError("production featurization requires a PolarMACE backbone")
        self.backbone = backbone
        self.backbone.to(dtype=dtype)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.checkpoint_sha256 = checkpoint_sha256
        self.mace_version = mace_version
        self.model_id = model_id
        self.feature_mode = feature_mode
        self.dtype = dtype
        self.physics_config = physics_config or PhysicsConfig()
        self.cache = cache
        self.graph_builder = graph_builder
        self.private_adapter = private_adapter
        self.multipole_contract = multipole_contract
        self.parity_atol = parity_atol
        self.last_private_parity_error = 0.0
        self.resolved_feature_schema: str | None = None
        supported = getattr(backbone, "atomic_numbers", torch.empty(0, dtype=torch.long))
        self.backbone_elements = tuple(int(value) for value in supported.tolist())
        self.supported_elements = frozenset(
            value for value in self.backbone_elements if value in ALLOWED_ELEMENTS
        )
        if feature_mode == "all-scalars+norms" and private_adapter is None:
            self.private_adapter = PolarMACEPrivateLayerAdapter(mace_version)

    def train(self, mode: bool = True):
        """Train completion heads around this module without unfreezing MACE."""

        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def metadata(self) -> dict[str, Any]:
        """Return reconstruction metadata suitable for checkpoint manifests."""

        return {
            "model_id": self.model_id,
            "mace_version": self.mace_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dtype": str(self.dtype),
            "supported_elements": tuple(sorted(self.supported_elements)),
            "feature_mode": self.feature_mode,
            "private_adapter": getattr(self.private_adapter, "version", None),
            "feature_schema": self.resolved_feature_schema,
            "multipole_contract": self.multipole_contract,
        }

    @property
    def schema_identity(self) -> str:
        adapter = getattr(self.private_adapter, "version", "public")
        return (
            f"{self.model_id}:mace={self.mace_version}:mode={self.feature_mode}:"
            f"adapter={adapter}"
        )

    def _default_graph_builder(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        import numpy as np
        from mace.data import AtomicData, Configuration
        from mace.tools import AtomicNumberTable
        from mace.tools.torch_geometric import DataLoader

        properties = {
            "total_charge": float(total_charge.item()),
            "total_spin": float(total_spin.item()),
            "external_field": np.zeros(3),
            "fermi_level": 0.0,
        }
        config = Configuration(
            atomic_numbers=atomic_numbers.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy(),
            properties=properties,
            property_weights={},
        )
        z_table = AtomicNumberTable(self.backbone_elements)
        graph = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(self.backbone.r_max),
            heads=getattr(self.backbone, "heads", ["Default"]),
        )
        batch = next(iter(DataLoader([graph], batch_size=1, shuffle=False)))
        result = batch.to_dict()
        result["atomic_numbers"] = atomic_numbers
        return result

    def _build_graph(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        builder = self.graph_builder or self._default_graph_builder
        graph = builder(
            positions,
            atomic_numbers,
            total_charge,
            total_spin,
            self.dtype,
        )
        device = self._backbone_device()
        converted = {}
        for name, value in graph.items():
            if not torch.is_tensor(value):
                converted[name] = value
            elif torch.is_floating_point(value):
                converted[name] = value.to(device=device, dtype=self.dtype)
            else:
                converted[name] = value.to(device=device)
        return converted

    def _backbone_device(self) -> torch.device:
        for tensor in list(self.backbone.parameters()) + list(self.backbone.buffers()):
            return tensor.device
        return torch.device("cpu")

    @staticmethod
    def _scalars_and_norms(hidden: torch.Tensor, irreps_string: str) -> torch.Tensor:
        o3 = _e3nn_o3()

        irreps = o3.Irreps(irreps_string)
        values = []
        for (multiplicity, irrep), feature_slice in zip(irreps, irreps.slices()):
            block = hidden[:, feature_slice].reshape(
                hidden.shape[0], multiplicity, irrep.dim
            )
            if irrep.l == 0:
                values.append(block[..., 0])
            else:
                values.append(torch.linalg.vector_norm(block, dim=-1))
        return torch.cat(values, dim=-1)

    def _runtime_features(
        self,
        graph: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
    ) -> MACEAtomicFeatures:
        public_scalars = outputs["node_feats"].to(dtype=self.dtype)
        natom = public_scalars.shape[0]
        equivariant = public_scalars.new_zeros((natom, 0))
        schema_suffix = "public"
        layer_count = 1
        if self.feature_mode == "all-scalars+norms":
            private = self.private_adapter.extract(self.backbone, graph, outputs)
            difference = (private.final_scalars - public_scalars).abs().max()
            self.last_private_parity_error = float(difference.detach().cpu())
            if not torch.allclose(
                private.final_scalars,
                public_scalars,
                atol=self.parity_atol,
                rtol=1.0e-6,
            ):
                raise RuntimeError(
                    "private PolarMACE adapter failed public-final-scalar parity"
                )
            invariant = torch.cat(
                (
                    public_scalars,
                    self._scalars_and_norms(private.hidden, private.hidden_irreps),
                ),
                dim=-1,
            )
            equivariant = private.hidden
            schema_suffix = (
                f"private={private.adapter_version}:irreps={private.hidden_irreps}"
            )
            layer_count = private.layer_count
        else:
            invariant = public_scalars
        feature_schema = (
            f"{self.schema_identity}:inv={invariant.shape[1]}:"
            f"equiv={equivariant.shape[1]}:layers={layer_count}:{schema_suffix}"
        )
        if self.resolved_feature_schema not in {None, feature_schema}:
            raise RuntimeError("PolarMACE runtime feature schema changed within a run")
        self.resolved_feature_schema = feature_schema
        return MACEAtomicFeatures(
            invariant=invariant.detach(),
            equivariant=equivariant.detach(),
            batch=torch.zeros(natom, dtype=torch.long, device=invariant.device),
            atomic_numbers=atomic_numbers.to(invariant.device),
            total_charge=total_charge.to(invariant),
            total_spin=total_spin.to(invariant),
            feature_schema=feature_schema,
        )

    def _run_single(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
    ) -> tuple[MACEAtomicFeatures, PolarMACEDirectOutputs]:
        graph = self._build_graph(
            positions, atomic_numbers, total_charge, total_spin
        )
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(graph, training=False, compute_force=False)
            features = self._runtime_features(
                graph, outputs, atomic_numbers, total_charge, total_spin
            )
        required = {"density_coefficients", "charges", "dipole"}
        missing = required.difference(outputs)
        if missing:
            raise ValueError(f"PolarMACE output is missing {sorted(missing)}")
        density = outputs["density_coefficients"].detach().to(dtype=self.dtype)
        if density.ndim != 2 or density.shape[1] != 4:
            raise ValueError(
                "PolarMACE artifact has an incompatible direct multipole width"
            )
        direct = PolarMACEDirectOutputs(
            density_coefficients=density,
            charges=outputs["charges"].detach().to(dtype=self.dtype),
            molecular_dipole_eangstrom=outputs["dipole"].detach().to(dtype=self.dtype),
            positions_angstrom=graph["positions"].detach().to(dtype=self.dtype),
            batch=torch.zeros(
                atomic_numbers.numel(), dtype=torch.long, device=density.device
            ),
            total_charge=total_charge.to(density),
            multipole_contract=self.multipole_contract,
        )
        reconstructed = (
            direct.charges[:, None] * direct.positions_angstrom
            + direct.intrinsic_dipole_eangstrom
        ).sum(0, keepdim=True)
        if not torch.allclose(
            reconstructed,
            direct.molecular_dipole_eangstrom,
            atol=1.0e-5,
            rtol=1.0e-6,
        ):
            raise ValueError(
                "PolarMACE direct coefficients do not reconstruct molecular dipole"
            )
        return features, direct

    def _cache_key(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
    ) -> str:
        return MACEFeatureCacheKey.from_tensors(
            checkpoint_sha256=self.checkpoint_sha256,
            mace_version=self.mace_version,
            feature_schema=self.schema_identity,
            physics_config_hash=self.physics_config.physics_hash,
            atomic_numbers=atomic_numbers,
            coordinates_angstrom=positions,
            total_charge=float(total_charge.item()),
            total_spin=float(total_spin.item()),
            dtype=self.dtype,
        ).cache_hash

    def forward_monomer(
        self,
        positions_angstrom: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> tuple[MACEAtomicFeatures, PolarMACEDirectOutputs]:
        """Run one or more isolated monomers and preserve input atom ordering."""

        if positions_angstrom.ndim != 2 or positions_angstrom.shape[1] != 3:
            raise ValueError("positions must have shape [n_atom, 3]")
        if positions_angstrom.shape[0] == 0:
            raise ValueError("at least one atom is required")
        if not torch.is_floating_point(positions_angstrom) or not torch.isfinite(
            positions_angstrom
        ).all():
            raise ValueError("positions must be finite floating values")
        if atomic_numbers.shape != (positions_angstrom.shape[0],) or atomic_numbers.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("atomic numbers must be a rank-1 integer tensor")
        if atomic_numbers.device != positions_angstrom.device:
            raise ValueError("atomic numbers and positions must share a device")
        for name, values in (("total_charge", total_charge), ("total_spin", total_spin)):
            if not torch.is_floating_point(values) or not torch.isfinite(values).all():
                raise ValueError(f"{name} must contain finite floating values")
        unsupported = sorted(set(int(z) for z in atomic_numbers.tolist()) - self.supported_elements)
        if unsupported:
            raise ValueError(f"unsupported element(s) for PolarMACE: {unsupported}")
        if batch is None:
            batch = torch.zeros(
                atomic_numbers.numel(), dtype=torch.long, device=atomic_numbers.device
            )
        if batch.shape != atomic_numbers.shape or batch.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("batch must be a rank-1 integer tensor")
        if batch.device != positions_angstrom.device:
            raise ValueError("batch and monomer tensors must share a device")
        if (
            total_charge.device != positions_angstrom.device
            or total_spin.device != positions_angstrom.device
        ):
            raise ValueError("charge/spin and monomer tensors must share a device")
        monomers = sorted(int(value) for value in batch.unique().tolist())
        if monomers != list(range(len(monomers))):
            raise ValueError("monomer batch indices must be contiguous from zero")
        if total_charge.shape != (len(monomers),) or total_spin.shape != (
            len(monomers),
        ):
            raise ValueError("charge and spin must contain one value per monomer")

        per_monomer = []
        for monomer in monomers:
            atom_indices = torch.where(batch == monomer)[0]
            positions = positions_angstrom[atom_indices]
            numbers = atomic_numbers[atom_indices]
            charge = total_charge[monomer : monomer + 1]
            spin = total_spin[monomer : monomer + 1]
            key = self._cache_key(positions, numbers, charge, spin)
            if self.cache is not None and key in self.cache:
                features, direct = self.cache[key]
                features = _clone_features(features)
                direct = _clone_direct(direct)
                features = MACEAtomicFeatures(
                    invariant=features.invariant.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    equivariant=features.equivariant.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    batch=features.batch.to(device=positions.device),
                    atomic_numbers=features.atomic_numbers.to(
                        device=positions.device
                    ),
                    total_charge=features.total_charge.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    total_spin=features.total_spin.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    feature_schema=features.feature_schema,
                )
                direct = PolarMACEDirectOutputs(
                    density_coefficients=direct.density_coefficients.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    charges=direct.charges.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    molecular_dipole_eangstrom=(
                        direct.molecular_dipole_eangstrom.to(
                            device=positions.device, dtype=self.dtype
                        )
                    ),
                    positions_angstrom=direct.positions_angstrom.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    batch=direct.batch.to(device=positions.device),
                    total_charge=direct.total_charge.to(
                        device=positions.device, dtype=self.dtype
                    ),
                    multipole_contract=direct.multipole_contract,
                )
                result = (features, direct)
            else:
                if self.cache is not None and getattr(
                    self.cache, "strict_read_only", False
                ):
                    raise KeyError(f"prepared feature cache miss: {key}")
                result = self._run_single(positions, numbers, charge, spin)
                if self.cache is not None:
                    self.cache[key] = (
                        _clone_features(result[0]),
                        _clone_direct(result[1]),
                    )
            per_monomer.append((atom_indices, *result))

        first_features = per_monomer[0][1]
        first_direct = per_monomer[0][2]
        if self.resolved_feature_schema not in {
            None,
            first_features.feature_schema,
        }:
            raise RuntimeError("cached PolarMACE feature schema does not match runtime")
        self.resolved_feature_schema = first_features.feature_schema
        natom = atomic_numbers.numel()
        invariant = first_features.invariant.new_empty(
            (natom, first_features.invariant.shape[1])
        )
        equivariant = first_features.equivariant.new_empty(
            (natom, first_features.equivariant.shape[1])
        )
        density = first_direct.density_coefficients.new_empty((natom, 4))
        charges = first_direct.charges.new_empty(natom)
        positions = first_direct.positions_angstrom.new_empty((natom, 3))
        dipoles = []
        for atom_indices, features, direct in per_monomer:
            indices = atom_indices.to(invariant.device)
            invariant.index_copy_(0, indices, features.invariant)
            equivariant.index_copy_(0, indices, features.equivariant)
            density.index_copy_(0, indices, direct.density_coefficients)
            charges.index_copy_(0, indices, direct.charges)
            positions.index_copy_(0, indices, direct.positions_angstrom)
            dipoles.append(direct.molecular_dipole_eangstrom)
        features = MACEAtomicFeatures(
            invariant=invariant,
            equivariant=equivariant,
            batch=batch.to(invariant.device),
            atomic_numbers=atomic_numbers.to(invariant.device),
            total_charge=total_charge.to(invariant),
            total_spin=total_spin.to(invariant),
            feature_schema=first_features.feature_schema,
        )
        direct = PolarMACEDirectOutputs(
            density_coefficients=density,
            charges=charges,
            molecular_dipole_eangstrom=torch.cat(dipoles, dim=0),
            positions_angstrom=positions,
            batch=batch.to(density.device),
            total_charge=total_charge.to(density),
            multipole_contract=self.multipole_contract,
        )
        return features, direct

    def forward_dimer(self, batch: Any):
        """Run A and B as separate graph batches with shared frozen weights."""

        features_a, direct_a = self.forward_monomer(
            batch.RA,
            batch.ZA,
            batch.total_charge_A,
            batch.total_spin_A,
            batch=batch.molecule_ind_A,
        )
        features_b, direct_b = self.forward_monomer(
            batch.RB,
            batch.ZB,
            batch.total_charge_B,
            batch.total_spin_B,
            batch=batch.molecule_ind_B,
        )
        return features_a, direct_a, features_b, direct_b
