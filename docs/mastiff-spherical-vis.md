# S66×8 spherical-harmonic explorer

Open [`mastiff-spherical-vis.html`](mastiff-spherical-vis.html) in a WebGL-enabled
browser. The HTML embeds all 528 geometries; pinned CDN dependencies require
internet access. No Python server or model checkpoint is required.

- Click a dimer name; select any of its eight separation points.
- Click atoms in 3D, or their keyboard-accessible buttons, to toggle lobes.
  Every newly selected dimer starts with all lobes hidden.
- Select any real Racah l=1,2 channel. Teal/orange encode positive/negative
  values; radial distance is `scale × abs(value)`, not an energy or density.
- Monomer A/B have blue/red expanded back-face silhouette outlines around
  element-colored atoms and bonds. Intermonomer bonds are never inferred.
- Equivalent-frame mode averages values and masks sine channels, matching
  `AveragedFrameHarmonics`. Inspection mode enables sine channels for full
  frames and lets you select individual tied reference assignments.
- Axial sites support only 10/20. Zero or masked channels are reported.
  Axes are drawn only for selected atoms; they can be hidden separately.

## Frame assignment limitation

This is a **visualization heuristic**, not an audited MASTIFF chemical
preprocessor. Bonds are inferred separately in each monomer at 1.0 Rₑ using
1.2 times the sum of covalent radii. Iterative graph colors rank reference
roles, retaining every tied assignment. Those plans are held fixed across all
eight separations. Full frames use the branch's z-then-x convention. Terminal
and wholly linear sites are axial; disconnected sites are explicitly isotropic.
Mixed degenerate reference sets fail instead of silently selecting a lab axis.

The default average is invariant to reference-row order; an individual
inspection frame is deliberately not a permutation-invariant prediction.
No fitted coefficients or predicted energies are shown. Reference plans and
the source hash are inspectable in the HTML.

## Rebuild and test

```bash
python scripts/build_mastiff_spherical_vis.py \
  --source /path/to/psi4/psi4/share/psi4/databases/S66by8.py
python -m pytest tests/test_spherical_vis.py -q
node --test tests/test_spherical_vis.cjs
cp docs/mastiff-spherical-vis.html ~/docs/mastiff-spherical-vis.html
```

The builder parses literal database strings without executing the source.
The source SHA-256 is embedded for provenance. Geometry source: Psi4's
LGPL-3.0 S66by8 database, citing Řezáč et al., JCTC 7, 2427 (2011).
Unlike the standalone S66 database, these are the actual S66×8 geometries.

Source assets are under `scripts/spherical_vis/`; edit these, then rebuild.
The page uses Lavish's DaisyUI/Tailwind fallback design since this Python
project has no existing web design system.

## Trained-model inspector

The same builder accepts a verified experiment export:

```bash
python scripts/build_mastiff_spherical_vis.py \
  --model-data /path/to/models.json \
  --output /path/to/experiment/docs/mastiff-spherical-vis.html
MASTIFF_MODEL_DATA=/path/to/models.json \
  node --test tests/test_spherical_model_export.cjs
```

Keep checkpoint binaries, model-derived exports and the trained-model HTML in
the paired `qcmlforge-exp` worktree, not in the public source repository.
For this campaign, the exporter is
`analysis/spherical-vis/export_models.py` in the `mastiff-exch` experiment
worktree. It imports the experiment's authoritative `Arm`, chemical perception,
bounded-coefficient and frame-harmonic APIs. The original basis-only viewer
remains available with `--source`.

The model export supplies all 528 geometries, actual reference plans, chemical
types, physical A/B/coefficient values, pair energies, total/reference exchange,
minimum angular factors, overlap flags and source/checkpoint hashes. The browser
reconstructs numerical equation terms for inspection; plots use exported pair
energies. JavaScript parity tests verify every reconstructed pair against those
authoritative values, including B's opposite outgoing direction.

Surfaces default to the learned angular factor `f`, with model overlay and
difference views. Isotropic `f=1` is a grey sphere. Per-type tables show effective
model mode and per-degree coefficient budgets. The separate atom-pair viewer
selects one atom from each monomer. Pair plots use actual atom-pair distances
across the eight original geometries; dimer plots separately compare reference
exchange and CLIFF2 totals. **There are no SAPT atom-pair reference labels here.**
The dashed pair curve sets angular factors to one with the same radial
parameters; it is not a separately trained isotropic model.

Frame inspection never changes predicted energies: real models use two-level
orbit averaging, and the one-reference option is only a display diagnostic.
The scalar fields are encoded as exact degree-2 Cartesian polynomials sampled
from the Python frame API, not from the preliminary graph heuristic. v4 axial
uses only C10/C20; v5 C2v can also use C22c. Missing checkpoints are explicitly
unavailable. Use `--checkpoint NAME=/trusted/path/checkpoint-*.pt` on the exporter
to include an intermediate or v5 checkpoint, then rebuild or use the browser's
JSON import. Import resets the page and does not modify the saved HTML.

The **Harmonic display** selector distinguishes:

- **Plain basis:** all eight unweighted channels in available geometric frames,
  without model channel/sine masks. Equivalent-frame averaging may cancel
  channels; one-reference inspection exposes individual contributions. Axial
  sites still do not receive an invented transverse gauge.
- **Model-used basis:** only channels used by the chosen model, unweighted.
- **Scaled contribution:** the signed learned `a_lm × C_lm` term. This does not
  include the constant 1 of the full angular factor.

Selecting a display mode from the full-factor view switches to C20; select any
other channel in the harmonic dropdown. Full `f_i` remains a separate choice
there. Plain mode uses the full available geometric frames, even when inspecting
a v4 axial model, and disables model surface comparison because it has no
model-dependent coefficients. None of these controls changes pairwise energies.

Equation substitutions use two decimals **only for display**; calculation and
export retain full float64 precision. S66 is not a blind holdout: 112 records
share training pairs; the correctly reconstructed pair-disjoint slice has 416.
The `novel-mask.json` in the historical report is monomer novelty, **not** a
pair-disjoint mask.

## Remote Lavish review

On the remote machine:

```bash
LAVISH_AXI_HOST=127.0.0.1 npx -y lavish-axi \
  ~/docs/mastiff-spherical-vis.html --no-open
```

On your local device, substitute your usual SSH destination:

```bash
ssh -fN -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:4387:127.0.0.1:4387 YOUR_SSH_HOST
```

Open the returned session URL using `http://127.0.0.1:4387` as the origin.
Do not bind the unauthenticated Lavish server to a public interface.
