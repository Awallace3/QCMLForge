"""Shared low-level model and lightweight training harnesses for MACE/AP3D3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch

from .long_range import LongRangeSAPTProvider, assemble_sapt_components
from .pair import MACEPairResidualCore
from .schema import (
    COMPONENT_ORDER,
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    InductionDiagnostics,
)


MACE_AP3D3_ARCHITECTURES = {
    "direct-polar": {
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
        "provider_kind": "direct",
    },
    "hybrid-h1": {
        "pair_mode": "h1",
        "feature_mode": "final-layer-scalars",
        "provider_kind": "legacy",
    },
    "hybrid-h2": {
        "pair_mode": "h2",
        "feature_mode": "all-scalars+norms",
        "provider_kind": "legacy",
    },
    "atomhead": {
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
        "provider_kind": "atomhead",
    },
}

_PROVIDER_CLASS_KINDS = {
    "PolarDirectPropertyProvider": "direct",
    "LegacyAtomMPNNPropertyProvider": "legacy",
    "MACEAtomPropertyModel": "atomhead",
}


def _provider_kind(provider: torch.nn.Module) -> str | None:
    explicit = getattr(provider, "provider_kind", None)
    if explicit is not None:
        return str(explicit)
    return _PROVIDER_CLASS_KINDS.get(type(provider).__name__)


def _freeze_backbone(featurizer: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(featurizer, "backbone", None)
    if not isinstance(backbone, torch.nn.Module):
        raise ValueError("MACE featurizer must expose its backbone module")
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


def _column_ledger(values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: values[:, index] for index, name in enumerate(COMPONENT_ORDER)}


@dataclass(frozen=True)
class MACEAP3D3Result:
    """Differentiable prediction plus separate residual and classical ledgers."""

    components: torch.Tensor
    residual: torch.Tensor
    classical: ClassicalEnergyBundle

    @property
    def component_ledger(self) -> Mapping[str, torch.Tensor]:
        return _column_ledger(self.components)

    @property
    def residual_ledger(self) -> Mapping[str, torch.Tensor]:
        return _column_ledger(self.residual)

    @property
    def classical_ledger(self) -> Mapping[str, torch.Tensor]:
        return {
            "elst": self.classical.dimer_elst,
            "indu": self.classical.dimer_ind,
            "disp": self.classical.dimer_disp,
        }

    @property
    def induction_diagnostics(self) -> InductionDiagnostics:
        return self.classical.induction_diagnostics


class MACEAP3D3(torch.nn.Module):
    """One route-independent MACE/AP3D3 interaction-energy pipeline."""

    external_backbone_state_prefixes = ("featurizer.backbone.",)
    atomic_property_schema = "ap3-atomic-properties-cartesian-v1"

    def __init__(
        self,
        *,
        architecture: str,
        featurizer: torch.nn.Module,
        property_provider: torch.nn.Module,
        pair_core: MACEPairResidualCore,
        long_range_provider: LongRangeSAPTProvider,
        use_precomputed_classical: bool = False,
    ) -> None:
        super().__init__()
        if architecture not in MACE_AP3D3_ARCHITECTURES:
            raise ValueError(f"unsupported MACE/AP3D3 architecture: {architecture}")
        expected = MACE_AP3D3_ARCHITECTURES[architecture]
        actual_provider = _provider_kind(property_provider)
        if actual_provider != expected["provider_kind"]:
            raise ValueError(
                f"{architecture} property provider must be "
                f"{expected['provider_kind']}, got {actual_provider}"
            )
        if getattr(pair_core, "pair_mode", None) != expected["pair_mode"]:
            raise ValueError(
                f"{architecture} pair topology must be {expected['pair_mode']}"
            )
        if getattr(pair_core, "feature_mode", None) != expected["feature_mode"]:
            raise ValueError(
                f"{architecture} pair feature mode must be "
                f"{expected['feature_mode']}"
            )
        allowed_pair_ids = {architecture}
        if architecture == "hybrid-h1":
            allowed_pair_ids.add("MACE-AP3D3-H1")
        elif architecture == "hybrid-h2":
            allowed_pair_ids.add("MACE-AP3D3-H2")
        if getattr(pair_core, "architecture_id", None) not in allowed_pair_ids:
            raise ValueError(
                f"pair architecture identifier must match {architecture}"
            )
        if getattr(featurizer, "feature_mode", None) != expected["feature_mode"]:
            raise ValueError(
                f"{architecture} featurizer feature mode must be "
                f"{expected['feature_mode']}"
            )
        self.architecture = architecture
        self.featurizer = featurizer
        self.property_provider = property_provider
        self.pair_core = pair_core
        self.long_range_provider = long_range_provider
        self.use_precomputed_classical = bool(use_precomputed_classical)
        physics = getattr(long_range_provider, "config", None)
        core_cutoff = getattr(getattr(pair_core, "ap3_core", None), "r_cut_im", None)
        if physics is not None and core_cutoff != physics.neural_cutoff:
            raise ValueError(
                "PhysicsConfig.neural_cutoff must equal the AP3 r_cut_im "
                "residual support"
            )
        _freeze_backbone(featurizer)
        self.last_residual_ledger: dict[str, torch.Tensor] = {}
        self.last_classical_ledger: dict[str, torch.Tensor] = {}
        self.last_induction_diagnostics: InductionDiagnostics | None = None

    @property
    def no_disp_nn(self) -> bool:
        explicit = getattr(self.pair_core, "no_disp_nn", None)
        if explicit is not None:
            return bool(explicit)
        ap3_core = getattr(self.pair_core, "ap3_core", None)
        return bool(getattr(ap3_core, "no_disp_nn", False))

    def train(self, mode: bool = True):
        super().train(mode)
        self.featurizer.backbone.eval()
        return self

    def create_checkpoint_v3(
        self,
        *,
        config: Mapping[str, Any],
        external_mace: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a state-only v3 record without foundation tensors."""

        from apnet_pt import model_io

        return model_io.create_mace_checkpoint_v3(
            self,
            config=config,
            external_mace=external_mace,
            metadata=metadata,
        )

    def save_checkpoint_v3(
        self,
        path: str,
        *,
        config: Mapping[str, Any],
        external_mace: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save a v3 record while keeping the MACE backbone external."""

        from apnet_pt import model_io

        checkpoint = self.create_checkpoint_v3(
            config=config,
            external_mace=external_mace,
            metadata=metadata,
        )
        model_io.save_checkpoint(checkpoint, path)

    @staticmethod
    def load_checkpoint_v3(
        checkpoint_path: str,
        *,
        mace_artifact_path: str,
        model_factory: Callable[[Mapping[str, Any], torch.nn.Module], torch.nn.Module],
        backbone_loader: Callable[..., torch.nn.Module],
        semantic_expectations: Mapping[str, Any] | None = None,
        constructor_overrides: Mapping[str, Any] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "MACEAP3D3":
        """Reconstruct from v3 metadata and a verified local artifact."""

        from apnet_pt import model_io

        model = model_io.load_mace_checkpoint_v3(
            checkpoint_path,
            mace_artifact_path=mace_artifact_path,
            model_factory=model_factory,
            backbone_loader=backbone_loader,
            semantic_expectations=semantic_expectations,
            constructor_overrides=constructor_overrides,
            map_location=map_location,
        )
        if not isinstance(model, MACEAP3D3):
            raise TypeError("v3 model factory must reconstruct MACEAP3D3")
        return model

    def forward(
        self,
        batch: Any,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | MACEAP3D3Result:
        features_a, direct_a, features_b, direct_b = self.featurizer.forward_dimer(
            batch
        )
        props_a, props_b = self.property_provider(
            batch,
            features_a,
            features_b,
            direct_a=direct_a,
            direct_b=direct_b,
        )
        residual = self.pair_core(
            batch,
            features_a,
            features_b,
            props_a,
            props_b,
        )
        if self.use_precomputed_classical:
            classical = getattr(batch, "precomputed_classical", None)
            if not isinstance(classical, ClassicalEnergyBundle):
                raise ValueError(
                    "requested precomputed classical path requires a "
                    "ClassicalEnergyBundle on the batch"
                )
            physics = getattr(self.long_range_provider, "config", None)
            expected_hash = getattr(physics, "physics_hash", None)
            if (
                not expected_hash
                or classical.physics_config_hash != expected_hash
            ):
                raise ValueError(
                    "precomputed classical bundle physics hash does not match "
                    "the active PhysicsConfig"
                )
        else:
            classical = self.long_range_provider(batch, props_a, props_b)
        components = assemble_sapt_components(
            residual,
            classical,
            no_disp_nn=self.no_disp_nn,
        )
        expected_shape = (int(batch.total_charge_A.numel()), len(COMPONENT_ORDER))
        if components.shape != expected_shape:
            raise RuntimeError(
                f"public SAPT predictions must have shape {expected_shape}"
            )
        if not torch.isfinite(components).all():
            raise RuntimeError("public SAPT predictions contain non-finite values")
        result = MACEAP3D3Result(components, residual, classical)
        self.last_residual_ledger = {
            name: value.detach() for name, value in result.residual_ledger.items()
        }
        self.last_classical_ledger = {
            name: value.detach() for name, value in result.classical_ledger.items()
        }
        self.last_induction_diagnostics = result.induction_diagnostics
        return result if return_details else components


class MACEAP3D3Model:
    """Small batch-oriented optimization and prediction harness."""

    def __init__(
        self,
        model: MACEAP3D3,
        *,
        include_total_mse: bool = False,
    ) -> None:
        self.model = model
        self.include_total_mse = include_total_mse
        self.last_component_losses: dict[str, torch.Tensor] = {}
        self.last_total_energy_loss: torch.Tensor | None = None
        self.last_loss: torch.Tensor | None = None

    @staticmethod
    def _validate_target(prediction: torch.Tensor, target: torch.Tensor) -> None:
        if target.shape != prediction.shape:
            raise ValueError("SAPT labels must retain full [n_dimer, 4] shape")
        if not torch.is_floating_point(target) or not torch.isfinite(target).all():
            raise ValueError("SAPT labels must be finite floating values")

    def compute_loss(
        self,
        batch: Any,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self.model(batch)
        if target is None:
            if not hasattr(batch, "y"):
                raise ValueError("batch does not contain SAPT labels")
            target = batch.y
        self._validate_target(prediction, target)
        component_losses = {
            name: (prediction[:, index] - target[:, index]).square().mean()
            for index, name in enumerate(COMPONENT_ORDER)
        }
        loss_terms = list(component_losses.values())
        total_energy_loss = None
        if self.include_total_mse:
            total_energy_loss = (
                prediction.sum(dim=1) - target.sum(dim=1)
            ).square().mean()
            loss_terms.append(total_energy_loss)
        loss = torch.stack(loss_terms).mean()
        self.last_total_energy_loss = (
            total_energy_loss.detach() if total_energy_loss is not None else None
        )
        if not torch.isfinite(loss):
            raise RuntimeError("MACE/AP3D3 loss is non-finite")
        return loss, component_losses

    def train_step(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        *,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, component_losses = self.compute_loss(batch, target)
        loss.backward()
        optimizer.step()
        self.last_component_losses = {
            name: value.detach() for name, value in component_losses.items()
        }
        self.last_loss = loss.detach()
        return self.last_loss

    def fit_epoch(
        self,
        batches: Iterable[Any],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        losses = [float(self.train_step(batch, optimizer)) for batch in batches]
        if not losses:
            raise ValueError("training epoch requires at least one batch")
        return sum(losses) / len(losses)

    def predict_batch(self, batch: Any) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(batch).detach()
        self.model.train(was_training)
        return prediction

    def predict(self, batches: Iterable[Any]) -> torch.Tensor:
        predictions = [self.predict_batch(batch) for batch in batches]
        if not predictions:
            raise ValueError("prediction requires at least one batch")
        return torch.cat(predictions, dim=0)


class MACEAtomicPropertiesModel(torch.nn.Module):
    """Frozen-feature harness for direct-completion or learned atom heads."""

    _PROPERTY_MODES = {
        "direct-completion": "direct",
        "learned": "atomhead",
    }

    def __init__(
        self,
        *,
        property_mode: str,
        featurizer: torch.nn.Module,
        property_provider: torch.nn.Module,
    ) -> None:
        super().__init__()
        if property_mode not in self._PROPERTY_MODES:
            raise ValueError(f"unsupported atomic property mode: {property_mode}")
        expected_provider = self._PROPERTY_MODES[property_mode]
        if _provider_kind(property_provider) != expected_provider:
            raise ValueError(
                f"{property_mode} requires {expected_provider} property provider"
            )
        self.property_mode = property_mode
        self.featurizer = featurizer
        self.property_provider = property_provider
        _freeze_backbone(featurizer)
        self.last_property_losses: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.featurizer.backbone.eval()
        return self

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> AtomicPropertyBundle:
        features, direct = self.featurizer.forward_monomer(
            positions,
            atomic_numbers,
            total_charge,
            total_spin,
            batch=batch,
        )
        if self.property_mode == "direct-completion":
            return self.property_provider.forward_monomer(features, direct)
        return self.property_provider.forward_monomer(features)

    @staticmethod
    def compute_loss(
        prediction: AtomicPropertyBundle,
        target: AtomicPropertyBundle,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = {
            name: (getattr(prediction, name) - getattr(target, name)).square().mean()
            for name in target.__dataclass_fields__
        }
        loss = torch.stack(tuple(losses.values())).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("atomic property loss is non-finite")
        return loss, losses

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train_step(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        *,
        target: AtomicPropertyBundle,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = self(
            positions,
            atomic_numbers,
            total_charge,
            total_spin,
            batch=batch,
        )
        loss, losses = self.compute_loss(prediction, target)
        loss.backward()
        optimizer.step()
        self.last_property_losses = {
            name: value.detach() for name, value in losses.items()
        }
        return loss.detach()
