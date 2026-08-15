"""Central fail-closed resolver for the external PolarMACE test artifact."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from apnet_pt.mace.encoder import POLAR_1S_SHA256

POLAR_1S_SIZE = 33_375_439


def polar_mace_artifact() -> Path:
    """Resolve and verify the environment-provided artifact before deserialization."""

    value = os.environ.get("QCMLFORGE_POLARMACE_ARTIFACT")
    required = os.environ.get("QCMLFORGE_REQUIRE_POLARMACE") == "1"
    if not value:
        if required:
            pytest.fail("QCMLFORGE_POLARMACE_ARTIFACT is required for protected CI-2")
        pytest.skip("QCMLFORGE_POLARMACE_ARTIFACT is not configured")
    artifact = Path(value)
    if not artifact.is_file():
        if required:
            pytest.fail("configured PolarMACE artifact is missing")
        pytest.skip("configured PolarMACE artifact is unavailable")
    if artifact.stat().st_size != POLAR_1S_SIZE:
        pytest.fail("configured PolarMACE artifact size does not match the canonical artifact")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != POLAR_1S_SHA256:
        pytest.fail("configured PolarMACE artifact digest does not match the canonical artifact")
    return artifact
