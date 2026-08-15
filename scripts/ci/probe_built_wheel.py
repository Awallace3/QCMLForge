#!/usr/bin/env python3
"""Probe an installed QCMLForge wheel from outside its source checkout."""

from __future__ import annotations

import argparse
from importlib import metadata, resources
import hashlib
import math
from pathlib import Path
import sys
import zipfile

POLARMACE_SHA256 = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"

REQUIRED = {
    "apnet_pt/mace/__init__.py", "apnet_pt/mace/encoder.py",
    "apnet_pt/mace/schema.py", "apnet_pt/mace/model.py",
    "apnet_pt/mace/pair.py", "apnet_pt/mace/properties.py",
    "apnet_pt/mace/long_range.py", "qcml_dftd3/__init__.py",
    "qcml_dftd3/d3.py", "qcml_dftd3/data/__init__.py",
    "qcml_dftd3/data/reference-c6.pt",
}


def inspect_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        digests = {}
        for name in names:
            digest = hashlib.sha256()
            with archive.open(name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[name] = digest.hexdigest()
    missing = REQUIRED - set(names)
    if missing:
        raise RuntimeError(f"wheel is missing required files: {sorted(missing)}")
    for name in names:
        lower = name.lower()
        if digests[name] == POLARMACE_SHA256:
            raise RuntimeError(
                f"wheel contains the forbidden foundation artifact digest: {name}"
            )
        if "__pycache__" in lower or lower.endswith((".pyc", ".pyo")):
            raise RuntimeError(f"wheel contains Python cache material: {name}")
        if lower.endswith(".model") or "polar" in Path(lower).name:
            raise RuntimeError(f"wheel contains a forbidden foundation artifact: {name}")
        if lower.endswith((".pt", ".pth", ".ckpt")) and name != "qcml_dftd3/data/reference-c6.pt":
            raise RuntimeError(f"wheel contains an unexpected checkpoint payload: {name}")


def probe_installed(checkout: Path | None) -> None:
    class BlockOptional:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mace" or fullname.startswith("mace.") or fullname == "e3nn" or fullname.startswith("e3nn."):
                raise ModuleNotFoundError(f"optional import blocked: {fullname}")
            return None
    sys.meta_path.insert(0, BlockOptional())
    import apnet_pt
    import apnet_pt.mace
    assert "mace" not in sys.modules and "e3nn" not in sys.modules
    requirements = metadata.requires("qcmlforge") or []
    unconditional = [value for value in requirements if "extra ==" not in value]
    assert not any(value.lower().startswith(("mace-torch", "e3nn", "graph-longrange")) for value in unconditional)
    resource = resources.files("qcml_dftd3.data").joinpath("reference-c6.pt")
    assert resource.is_file()
    import torch
    with resource.open("rb") as handle:
        value = torch.load(handle, map_location="cpu", weights_only=True)
    tensors = [item for item in ([value] if torch.is_tensor(value) else value.values()) if torch.is_tensor(item)]
    assert tensors and all(item.numel() and torch.isfinite(item).all() for item in tensors)
    if checkout:
        checkout = checkout.resolve()
        for module_path in (Path(apnet_pt.__file__).resolve(), Path(str(resource)).resolve()):
            if module_path.is_relative_to(checkout):
                raise RuntimeError(f"probe imported from checkout: {module_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args(argv)
    inspect_wheel(args.wheel)
    if args.installed:
        probe_installed(args.checkout)
    print("built-wheel probe PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
