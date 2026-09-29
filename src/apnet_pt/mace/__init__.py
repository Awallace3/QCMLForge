"""Optional MACE/AP3D3 integration.

The package exposes schema types eagerly, while adapters and models remain lazy
so base QCMLForge imports never require the optional ``mace`` dependency.
"""

from .schema import (
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    ExternalMACEArtifact,
    InductionDiagnostics,
    MACEAtomicFeatures,
    MACEFeatureCacheKey,
    PhysicsConfig,
    PolarMACEDirectOutputs,
)

__all__ = [
    "AtomicPropertyBundle",
    "ClassicalEnergyBundle",
    "ExternalMACEArtifact",
    "InductionDiagnostics",
    "MACEAtomicFeatures",
    "MACEFeatureCacheKey",
    "PhysicsConfig",
    "PolarMACEDirectOutputs",
    "MACEPolarFeaturizer",
    "PolarMACEPrivateLayerAdapter",
    "AtomicPropertyProvider",
    "MACEPropertyCompletionHeads",
    "MACEAtomPropertyModel",
    "PolarDirectPropertyProvider",
    "LegacyAtomMPNNPropertyProvider",
    "MACEPairResidualCore",
    "MACEExchangeCorrection",
    "MACEMASTIFFExchange",
    "MACEAP3D3",
    "MACEAP3D3Model",
    "MACEAP3D3Result",
    "MACEAtomicPropertiesModel",
]


def __getattr__(name):
    """Load optional adapters only when explicitly requested."""

    if name in {"MACEPolarFeaturizer", "PolarMACEPrivateLayerAdapter"}:
        from . import encoder

        return getattr(encoder, name)
    if name in {
        "AtomicPropertyProvider",
        "MACEPropertyCompletionHeads",
        "MACEAtomPropertyModel",
        "PolarDirectPropertyProvider",
        "LegacyAtomMPNNPropertyProvider",
    }:
        from . import properties

        return getattr(properties, name)
    if name == "MACEPairResidualCore":
        from .pair import MACEPairResidualCore

        return MACEPairResidualCore
    if name == "MACEExchangeCorrection":
        from .exchange import MACEExchangeCorrection

        return MACEExchangeCorrection
    if name == "MACEMASTIFFExchange":
        from .mastiff import MACEMASTIFFExchange

        return MACEMASTIFFExchange
    if name in {
        "MACEAP3D3",
        "MACEAP3D3Model",
        "MACEAP3D3Result",
        "MACEAtomicPropertiesModel",
    }:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
