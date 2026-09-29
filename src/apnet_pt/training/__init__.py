"""Training registries and factories with lazy optional-dependency boundaries."""

from .mace_ap3d3_factory import (
    MACE_AP3D3_OPTIONS,
    MACEFactoryDependencies,
    MACEFactoryResult,
    MACETrainingPlan,
    ResolvedMACEOption,
    build_mace_ap3d3_harness,
    build_mace_atomic_harness,
    dispatch_mace_cli,
    expected_resume_semantics,
    looks_like_mace_option,
    resolve_mace_option,
    validate_mace_cli_args,
)

__all__ = [
    "MACE_AP3D3_OPTIONS",
    "MACEFactoryDependencies",
    "MACEFactoryResult",
    "MACETrainingPlan",
    "ResolvedMACEOption",
    "build_mace_ap3d3_harness",
    "build_mace_atomic_harness",
    "dispatch_mace_cli",
    "expected_resume_semantics",
    "looks_like_mace_option",
    "resolve_mace_option",
    "validate_mace_cli_args",
]
