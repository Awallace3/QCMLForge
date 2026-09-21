"""The two checks only the real PolarMACE artifact can make.

Every other MACE test runs against ``tests.mace_stub_harness``, whose feature
widths are invented.  Two properties cannot be stubbed.  The first is the
widths themselves: ``POLAR_1S_INVARIANT_DIMS`` and
``POLAR_1S_EQUIVARIANT_CHANNELS`` exist so a caller can size a pair core
*before* it owns a featurizer, and a stub agreeing with them proves only that
two invented numbers match.  The second is that a v3 checkpoint of a model
holding the 33 MB backbone stays small -- the stub backbone is 800 kB, so the
size floor the exclusion defends is not visible against it.

Skipped unless ``QCMLFORGE_POLARMACE_ARTIFACT`` points at the licensed
checkpoint; see ``tests/mace_integration.py``.
"""

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.encoder import (
    POLAR_1S_EQUIVARIANT_CHANNELS,
    POLAR_1S_INVARIANT_DIMS,
    POLAR_1S_SHA256,
    SUPPORTED_MACE_VERSION,
    MACEPolarFeaturizer,
    PolarMACEPrivateLayerAdapter,
    load_verified_polar_mace,
)
from apnet_pt.mace.model import MACEAP3D3
from apnet_pt.mace.pair import DIRECTIONAL_DEGREES, MACEPairResidualCore
from tests.mace_integration import polar_mace_artifact
from tests.mace_stub_harness import (
    StubLongRangeProvider,
    StubPropertyProvider,
    _augment_batch,
    _batch,
)
from tests.test_mace_checkpoint_v3 import _config, _external_metadata

pytestmark = pytest.mark.mace_integration

# 1 + 3 + 5 + 7: the artifact carries one block per degree through l=3.
_EQUIVARIANT_DIMS_PER_CHANNEL = 16
# Restated rather than read off the loaded module, so that an upstream
# relocation of the class shows up here instead of silently changing what a
# v3 record claims its external backbone is.
_POLAR_BACKBONE_CLASS = "mace.modules.extensions.PolarMACE"


def _backbone():
    return load_verified_polar_mace(
        polar_mace_artifact(), expected_sha256=POLAR_1S_SHA256
    )


def _featurizer(backbone, feature_mode):
    adapter = (
        PolarMACEPrivateLayerAdapter(SUPPORTED_MACE_VERSION)
        if feature_mode == "all-scalars+norms"
        else None
    )
    return MACEPolarFeaturizer(
        backbone,
        checkpoint_sha256=POLAR_1S_SHA256,
        feature_mode=feature_mode,
        private_adapter=adapter,
    )


def _real_config(featurizer, route):
    """``_config`` with the stub MACE block replaced by the artifact's own.

    The v3 writer cross-checks the record against the live featurizer, so this
    is where the real ``feature_schema`` -- widths and irreps included -- gets
    written into the checkpoint instead of the harness's invented one.
    """

    config = _config(POLAR_1S_SHA256, route)
    config["mace"] = {
        "model_id": featurizer.model_id,
        "version": featurizer.mace_version,
        "sha256": POLAR_1S_SHA256,
        "feature_schema": featurizer.resolved_feature_schema,
        "feature_mode": featurizer.feature_mode,
    }
    return config


def _water():
    positions = torch.tensor(
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
    )
    return positions, torch.tensor([8, 1, 1])


@pytest.mark.parametrize("feature_mode", sorted(POLAR_1S_INVARIANT_DIMS))
def test_declared_widths_are_the_widths_the_artifact_emits(feature_mode):
    featurizer = _featurizer(_backbone(), feature_mode)
    positions, numbers = _water()
    features, direct = featurizer.forward_monomer(
        positions, numbers, torch.zeros(1), torch.ones(1)
    )

    natom = numbers.numel()
    assert features.invariant.shape == (natom, POLAR_1S_INVARIANT_DIMS[feature_mode])
    assert direct.charges.shape == (natom,)
    schema = features.feature_schema
    assert f"inv={POLAR_1S_INVARIANT_DIMS[feature_mode]}" in schema

    if feature_mode == "final-layer-scalars":
        # No private adapter, so no equivariant block to slice and nothing a
        # directional route could be built on.
        assert features.equivariant.shape == (natom, 0)
        assert schema.endswith(":public")
        return

    channels = POLAR_1S_EQUIVARIANT_CHANNELS
    assert features.equivariant.shape == (
        natom,
        channels * _EQUIVARIANT_DIMS_PER_CHANNEL,
    )
    assert (
        f"irreps={channels}x0e+{channels}x1o+{channels}x2e+{channels}x3o" in schema
    )


def test_v3_round_trip_leaves_the_licensed_backbone_out_of_the_file(tmp_path):
    route = "hybrid-h3l3"
    pair_mode = _config(POLAR_1S_SHA256, route)["pair_mode"]
    backbone = _backbone()
    featurizer = _featurizer(backbone, "all-scalars+norms")
    # Sized from the declared constants alone.  A wrong constant fails here
    # rather than in a shape error some layers deeper.
    pair_core = MACEPairResidualCore(
        APNet3D3_AtomType_MPNN(dimer_prop_model=None, use_precomputed_classical=True),
        mace_feature_dim=POLAR_1S_INVARIANT_DIMS["all-scalars+norms"],
        pair_mode=pair_mode,
        feature_mode="all-scalars+norms",
        mace_equivariant_dim=POLAR_1S_EQUIVARIANT_CHANNELS,
    )
    assert DIRECTIONAL_DEGREES[pair_mode] == 3
    model = MACEAP3D3(
        architecture=route,
        featurizer=featurizer,
        property_provider=StubPropertyProvider("legacy"),
        pair_core=pair_core,
        long_range_provider=StubLongRangeProvider(),
    )
    batch = _augment_batch(_batch())
    expected = model(batch).detach()

    path = tmp_path / "mace-ap3d3-v3.pt"
    model.save_checkpoint_v3(
        str(path),
        config=_real_config(featurizer, route),
        external_mace={
            **_external_metadata(POLAR_1S_SHA256),
            "model_id": featurizer.model_id,
            "model_class": _POLAR_BACKBONE_CLASS,
        },
    )
    # Not a ratio: the record accounts for the backbone by count while
    # carrying none of it.  ``external`` is what the stub suite can only
    # assert against its own 200k-element placeholder.
    written = torch.load(path, map_location="cpu", weights_only=False)
    assert not any(
        key.startswith("featurizer.backbone.")
        for key in written["model_state_dict"]
    )
    assert written["config"]["parameter_counts"]["external"] == sum(
        parameter.numel() for parameter in backbone.parameters()
    )
    assert path.stat().st_size < polar_mace_artifact().stat().st_size

    def factory(config, loaded_backbone):
        rebuilt = MACEAP3D3(
            architecture=config["architecture"],
            featurizer=_featurizer(loaded_backbone, config["mace"]["feature_mode"]),
            property_provider=StubPropertyProvider("legacy"),
            pair_core=MACEPairResidualCore(
                APNet3D3_AtomType_MPNN(
                    dimer_prop_model=None, use_precomputed_classical=True
                ),
                mace_feature_dim=POLAR_1S_INVARIANT_DIMS[
                    config["mace"]["feature_mode"]
                ],
                pair_mode=config["pair_mode"],
                feature_mode=config["mace"]["feature_mode"],
                mace_equivariant_dim=POLAR_1S_EQUIVARIANT_CHANNELS,
            ),
            long_range_provider=StubLongRangeProvider(),
        )
        rebuilt.featurizer.checkpoint_sha256 = config["mace"]["sha256"]
        return rebuilt

    def loader(path, *, map_location="cpu"):
        return load_verified_polar_mace(
            path, expected_sha256=POLAR_1S_SHA256, device=map_location
        )

    reloaded = MACEAP3D3.load_checkpoint_v3(
        str(path),
        mace_artifact_path=str(polar_mace_artifact()),
        model_factory=factory,
        backbone_loader=loader,
    )
    torch.testing.assert_close(reloaded(batch).detach(), expected, rtol=0, atol=0)
