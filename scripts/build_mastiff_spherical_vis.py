"""Build a single-file S66x8 local-frame viewer from Psi4's S66by8.py.

The source is parsed as text, never executed. Only NumPy is required.
Frame plans are visualization heuristics, not production MASTIFF atom typing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


RADII = {"H": .31, "C": .76, "N": .71, "O": .66}
NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}
DISTANCES = ["0.9", "0.95", "1.0", "1.05", "1.1", "1.25", "1.5", "2.0"]


def monomer_plan(symbols: list[str], xyz: np.ndarray) -> tuple[list, list]:
    """Infer a frozen bond graph and tied reference plans for one monomer.

    Covalent radii with a 1.2 multiplier define bonds. Iterated graph colors
    prioritize reference roles without breaking ties by atom index. All tied
    z/x assignments are retained. Entirely linear sites are axial; isolated
    sites explicitly have no anisotropy. This is NOT audited chemical typing.
    """
    radii = np.array([RADII[s] for s in symbols])
    delta = xyz[:, None] - xyz[None, :]
    distance = np.linalg.norm(delta, axis=-1)
    graph = (distance > .1) & (distance < 1.2 * (radii[:, None] + radii))
    bonds = np.column_stack(np.where(np.triu(graph, 1))).tolist()
    neighbors = [np.flatnonzero(row).tolist() for row in graph]
    colors = [NUMBERS[s] for s in symbols]
    for _ in range(len(symbols)):
        keys = [(colors[i], tuple(sorted(colors[j] for j in ns)))
                for i, ns in enumerate(neighbors)]
        ranking = {key: i for i, key in enumerate(sorted(set(keys)))}
        updated = [ranking[key] for key in keys]
        if updated == colors:
            break
        colors = updated
    plans = []
    for i, ns in enumerate(neighbors):
        if not ns:
            plans.append({"kind": "isotropic", "refs": []})
            continue
        zs = [j for j in ns if colors[j] == max(colors[k] for k in ns)]
        refs = []
        for z in zs:
            xs = [j for j in ns if j != z]
            if xs:
                refs.extend([[z, x] for x in xs
                             if colors[x] == max(colors[k] for k in xs)])
        if refs:
            zvec = xyz[np.array(refs)[:, 0]] - xyz[i]
            xvec = xyz[np.array(refs)[:, 1]] - xyz[i]
            sine = np.linalg.norm(np.cross(zvec, xvec), axis=-1)
            sine /= np.linalg.norm(zvec, axis=-1) * np.linalg.norm(xvec, axis=-1)
            if np.all(sine > 1e-6):
                plans.append({"kind": "full", "refs": refs})
                continue
            if np.any(sine > 1e-6):
                raise ValueError("Mixed degenerate reference set; explicit typing needed")
        plans.append({"kind": "axial", "refs": [[z] for z in zs]})
    return bonds, plans


def parse_database(text: str) -> list[dict]:
    """Extract all 528 literal Angstrom dimers and names without importing Psi4."""
    pattern = (r"GEOS\['%s-%s-%s' % \(dbse, '(\d+)-([\d.]+)', 'dimer'\)\]"
               r'\s*=\s*qcdb.Molecule\("""(.*?)"""\)')
    matches = re.findall(pattern, text, re.S)
    if len(matches) != 528:
        raise ValueError(f"Expected 528 S66x8 geometries, found {len(matches)}")
    records = {}
    for number, scale, geometry in matches:
        if "units Angstrom" not in geometry or geometry.count("--") != 1:
            raise ValueError("Expected two monomers in Angstrom")
        atoms, fragments = [], []
        for fragment, block in enumerate(geometry.split("--")):
            for line in block.splitlines():
                fields = line.split()
                if len(fields) == 4 and fields[0] in RADII:
                    atoms.append([fields[0], *map(float, fields[1:])])
                    fragments.append(fragment)
        if not atoms or set(fragments) != {0, 1}:
            raise ValueError("Empty monomer")
        records[(int(number), scale)] = (atoms, fragments)
    dimers = []
    for number in range(1, 67):
        name_match = re.search(
            rf"TAGL\['%s-%s'\s*% \(dbse, '{number}-1.0'\s*\)\]"
            r'\s*=\s*"""(.*?)"""', text,
        )
        if not name_match:
            raise ValueError(f"Missing name for dimer {number}")
        atoms, fragments = records[(number, "1.0")]
        symbols = [a[0] for a in atoms]
        xyz = np.array([a[1:] for a in atoms])
        split = fragments.index(1)
        bonds, plans = [], []
        for start, end in [(0, split), (split, len(atoms))]:
            local_bonds, local_plans = monomer_plan(symbols[start:end], xyz[start:end])
            bonds.extend([[a + start, b + start] for a, b in local_bonds])
            plans.extend([{**p, "refs": [[r + start for r in row] for row in p["refs"]]}
                          for p in local_plans])
        geometries = {}
        for scale in DISTANCES:
            scaled_atoms, scaled_fragments = records[(number, scale)]
            if [a[0] for a in scaled_atoms] != symbols or scaled_fragments != fragments:
                raise ValueError("Atom order differs across separation points")
            geometries[scale] = [a[1:] for a in scaled_atoms]
        dimers.append(dict(id=number, name=" ".join(name_match[1].split()),
                           symbols=symbols, split=split, bonds=bonds, plans=plans,
                           geometries=geometries))
    return dimers


def main() -> None:
    """Generate the portable HTML with embedded data, JS and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="Psi4 S66by8.py (basis-only viewer)")
    inputs.add_argument("--model-data", type=Path, help="Verified experiment model export JSON")
    parser.add_argument("--output", type=Path, default=Path("docs/mastiff-spherical-vis.html"))
    args = parser.parse_args()
    if args.model_data:
        data = json.loads(args.model_data.read_text())
        if data.get("schema") != "mastiff-spherical-models-v1" or len(data["dimers"]) != 66:
            raise ValueError("Expected complete mastiff-spherical-models-v1 export")
    else:
        source = args.source.read_bytes()
        data = {"dimers": parse_database(source.decode()),
                "source_sha256": hashlib.sha256(source).hexdigest()}
    assets = Path(__file__).with_name("spherical_vis")
    html = (assets / "index.html").read_text()
    html = html.replace("/*__CORE__*/", (assets / "harmonics.js").read_text())
    html = html.replace("/*__MODEL_MATH__*/", (assets / "model_math.js").read_text())
    html = html.replace("/*__INSPECTOR__*/", (assets / "inspector.js").read_text())
    html = html.replace("/*__APP__*/", (assets / "viewer.js").read_text())
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("<", r"\u003c"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output}: 66 dimers / 528 geometries ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
