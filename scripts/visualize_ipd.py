#!/usr/bin/env python3
"""Render point-induced-dipole animations for two benchmark dimers.

The script includes:

* the equilibrium water dimer (``01_Water-Water_1.00``) from the QCMLForge
  pytest data; and
* the strongly polarized compressed acetic-acid dimer
  (``20_AcOH-AcOH_0.90``) from the ACS Chicago S66x8 dataframe.

Each system is rendered to a separate directory. The animation has three
stages: permanent multipole dipoles, direct induced dipoles, and all mutual
SCF iterations. Blue arrows are permanent dipoles and orange arrows are
induced dipoles.

Examples
--------
Render both systems and assemble their default 11-second GIFs::

    python scripts/visualize_ipd.py

Render only the strongly polarized acetic-acid dimer::

    python scripts/visualize_ipd.py --system acoh-acoh-090

Rebuild both GIFs with different timing, without invoking PyMOL::

    python scripts/visualize_ipd.py --gif-only \
        --permanent-pause 4 --direct-pause 2 --mutual-duration 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
DEFAULT_CONVERGENCE_THRESHOLD = 1.0e-8
PERMANENT_COLOR = (0.10, 0.35, 0.90)  # blue
INDUCED_COLOR = (1.00, 0.38, 0.05)  # orange


@dataclass(frozen=True)
class DimerSystem:
    """All point-induced-dipole inputs for one molecular dimer."""

    key: str
    display_name: str
    geometry_bohr: np.ndarray
    elements: tuple[str, ...]
    fragment_split: int
    permanent_charges: np.ndarray
    permanent_dipoles: np.ndarray
    polarizabilities: np.ndarray
    permanent_scale: float
    induced_scale: float
    rotation_degrees: float = 0.0


WATER_DIMER = DimerSystem(
    key="water",
    display_name="Water dimer",
    # tests/dataset_data/water_dimer_pes3.pkl, 01_Water-Water_1.00
    geometry_bohr=np.array(
        [
            [-1.32695822, -0.10593854, 0.01878815],
            [-1.93166523, 1.60017431, -0.02171052],
            [0.48664427, 0.07959810, 0.00986248],
            [4.28756329, 0.04977558, 0.00096004],
            [4.99927500, -0.77864268, 1.44872529],
            [4.99104090, -0.85013652, -1.40764654],
        ],
        dtype=float,
    ),
    elements=("O", "H", "H", "O", "H", "H"),
    fragment_split=3,
    permanent_charges=np.array(
        [
            -0.90282657,
            0.45219405,
            0.45063254,
            -0.90516140,
            0.45258175,
            0.45257967,
        ]
    ),
    permanent_dipoles=np.array(
        [
            [-0.12099517, -0.19035041, 0.00497184],
            [-0.02343171, 0.01783004, -0.00038207],
            [0.02653300, -0.01378759, 0.00027520],
            [-0.14261352, 0.17416981, -0.00394689],
            [0.00157417, -0.00108209, 0.02947683],
            [0.00140431, -0.00255545, -0.02939500],
        ],
        dtype=float,
    ),
    polarizabilities=np.array(
        [
            8.38374595553467,
            0.4842211422539944,
            0.4977805639070765,
            8.388563748172823,
            0.4855270362311864,
            0.4855449542590184,
        ],
        dtype=float,
    ),
    # Short permanent arrows avoid passing through the bonded hydrogens.
    permanent_scale=3.0,
    induced_scale=9.0,
    rotation_degrees=-30.0,
)


ACOH_ACOH_DIMER = DimerSystem(
    key="acoh-acoh-090",
    # display_name="AcOH-AcOH (S66x8 0.90x)",
    display_name="AcOH-AcOH",
    # ~/presentations/26/ACS_Chicago_CDS/data/s66x8_amoeba_amber.pkl,
    # system_id == 20_AcOH-AcOH_0.90. This is the largest-magnitude
    # amoebaplus_pol_ml_ap3 entry in that dataframe (-9.6255 kcal/mol).
    geometry_bohr=np.array(
        [
            [-2.00633961, 2.45124042, 0.55191341],
            [-0.67682643, 4.29054495, 1.00498030],
            [-1.11362226, 0.17936857, 0.00715981],
            [0.76427263, 0.24136077, 0.03479334],
            [-4.83472785, 2.53705147, 0.55984520],
            [-5.47264304, 4.43606405, 0.97947593],
            [-5.54235750, 1.93203794, -1.27177727],
            [-5.55052618, 1.21870409, 1.96447817],
            [4.97095086, 2.09328058, 0.51016600],
            [3.64130486, 0.25398276, 0.05703311],
            [4.07819448, 4.36555415, 1.05398849],
            [2.20020631, 4.30338549, 1.02677549],
            [7.79932748, 2.00653696, 0.50645384],
            [8.43726039, 0.11431792, 0.05759570],
            [8.52025524, 3.34890947, -0.87204260],
            [8.50139156, 2.57854670, 2.35093115],
        ],
        dtype=float,
    ),
    elements=("C", "O", "O", "H", "C", "H", "H", "H") * 2,
    fragment_split=8,
    permanent_charges=np.array(
        [
            1.02858006,
            -0.72836253,
            -0.72187361,
            0.49144176,
            -0.59391953,
            0.18315452,
            0.17042881,
            0.17055040,
            1.02850393,
            -0.72832825,
            -0.72188029,
            0.49143011,
            -0.59388107,
            0.18315494,
            0.17051723,
            0.17048326,
        ]
    ),
    permanent_dipoles=np.array(
        [
            [0.06257848, -0.05071005, -0.01222165],
            [0.06826644, 0.15705931, 0.03856025],
            [-0.09258240, -0.07155048, -0.01781094],
            [0.02590577, -0.00314270, -0.00059127],
            [-0.04652578, 0.00322477, 0.00021121],
            [-0.00972592, 0.04848588, 0.01082770],
            [-0.01636164, -0.01578093, -0.04485183],
            [-0.01676862, -0.03299435, 0.03406812],
            [-0.06258357, 0.05078046, 0.01193059],
            [-0.06829384, -0.15708307, -0.03843150],
            [0.09259947, 0.07154121, 0.01788700],
            [-0.02590876, 0.00314088, 0.00059155],
            [0.04654407, -0.00316261, -0.00044769],
            [0.00972073, -0.04832700, -0.01151370],
            [0.01680228, 0.03365111, -0.03338312],
            [0.01631500, 0.01489615, 0.04517894],
        ],
        dtype=float,
    ),
    # Free-atom BG polarizabilities multiplied by the PBE0/aug-cc-pVTZ
    # Hirshfeld volume ratios stored in the same S66x8 row.
    polarizabilities=np.array(
        [
            6.4935397082,
            6.6527103580,
            6.3606550265,
            0.7274310788,
            13.4357966534,
            1.3995017174,
            1.4782191050,
            1.4795592645,
            6.4941021312,
            6.6527348851,
            6.3608774979,
            0.7275452513,
            13.4349839178,
            1.3993985953,
            1.4786966008,
            1.4787446997,
        ],
        dtype=float,
    ),
    # Use one common scale so permanent and induced lengths are quantitative.
    permanent_scale=5.0,
    induced_scale=5.0,
)

SYSTEMS = {system.key: system for system in (WATER_DIMER, ACOH_ACOH_DIMER)}


def calculate_induced_dipoles(
    system: DimerSystem,
    max_iterations: int = 200,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Calculate direct dipoles and the successive mutual SCF iterates."""
    from apnet_pt.multipole import T_cart_Thole_damping

    n_atoms = len(system.geometry_bohr)
    tensors = np.zeros((n_atoms, n_atoms, 3, 3), dtype=float)
    charge_fields = np.zeros((n_atoms, n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            _, t1, t2, _, _ = T_cart_Thole_damping(
                system.geometry_bohr[i],
                system.geometry_bohr[j],
                system.polarizabilities[i],
                system.polarizabilities[j],
                thole_damping_param,
            )
            charge_fields[i, j] = t1
            tensors[i, j] = t2

    direct = np.zeros((n_atoms, 3), dtype=float)
    fragments = (
        range(system.fragment_split),
        range(system.fragment_split, n_atoms),
    )
    target_source_pairs = (
        (fragments[0], fragments[1]),
        (fragments[1], fragments[0]),
    )
    for targets, sources in target_source_pairs:
        source_indices = np.fromiter(sources, dtype=int)
        for i in targets:
            charge_term = np.einsum(
                "ji,j->i",
                charge_fields[i, source_indices],
                system.permanent_charges[source_indices],
            )
            dipole_term = np.einsum(
                "jik,jk->i",
                tensors[i, source_indices],
                system.permanent_dipoles[source_indices],
            )
            direct[i] = system.polarizabilities[i] * (charge_term + dipole_term)

    induced = direct.copy()
    field_dipoles = system.permanent_dipoles.copy()
    history: list[np.ndarray] = []
    for _ in range(max_iterations):
        previous = induced.copy()
        mutual = system.polarizabilities[:, None] * np.einsum(
            "ijab,jb->ia", tensors, field_dipoles
        )
        induced = (1.0 - omega) * previous + omega * (direct + mutual)
        history.append(induced.copy())
        field_dipoles = induced
        if np.linalg.norm(induced - previous) < convergence_threshold:
            break

    return direct, history


def _infer_bonds(system: DimerSystem) -> list[tuple[int, int]]:
    """Infer intrafragment covalent bonds for the generated PDB."""
    covalent_radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
    coordinates = system.geometry_bohr * BOHR_TO_ANGSTROM
    bonds = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            same_fragment = (i < system.fragment_split) == (j < system.fragment_split)
            if not same_fragment:
                continue
            cutoff = 1.25 * (
                covalent_radii[system.elements[i]] + covalent_radii[system.elements[j]]
            )
            if np.linalg.norm(coordinates[i] - coordinates[j]) <= cutoff:
                bonds.append((i + 1, j + 1))
    return bonds


def _pdb_string(system: DimerSystem) -> str:
    coordinates = system.geometry_bohr * BOHR_TO_ANGSTROM
    lines = []
    for index, (element, xyz) in enumerate(
        zip(system.elements, coordinates), start=1
    ):
        residue = 1 if index <= system.fragment_split else 2
        lines.append(
            f"HETATM{index:5d} {element:>2s}   MOL {residue:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          "
            f"{element:>2s}"
        )
    for atom_i, atom_j in _infer_bonds(system):
        lines.append(f"CONECT{atom_i:5d}{atom_j:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _dipole_cgo(
    system: DimerSystem,
    vectors: np.ndarray,
    scale: float,
    color: tuple[float, float, float],
    shaft_radius: float = 0.045,
) -> list[float]:
    from pymol.cgo import CONE, CYLINDER

    coordinates = system.geometry_bohr * BOHR_TO_ANGSTROM
    objects: list[float] = []
    for origin, vector in zip(coordinates, vectors):
        end = origin + scale * vector
        length = float(np.linalg.norm(end - origin))
        if length < 0.025:
            continue
        direction = (end - origin) / length
        head_length = min(0.22, 0.38 * length)
        shaft_end = end - direction * head_length
        radius = min(shaft_radius, 0.20 * length)
        objects.extend(
            [
                CYLINDER,
                *origin,
                *shaft_end,
                radius,
                *color,
                *color,
                CONE,
                *shaft_end,
                *end,
                2.25 * radius,
                0.0,
                *color,
                *color,
                1.0,
                1.0,
            ]
        )
    return objects


def _set_stage(
    system: DimerSystem,
    induced: np.ndarray | None,
    permanent_scale: float,
    induced_scale: float,
) -> None:
    from pymol import cmd

    cmd.delete("permanent_dipoles")
    cmd.delete("induced_dipoles")
    cmd.load_cgo(
        _dipole_cgo(
            system,
            system.permanent_dipoles,
            permanent_scale,
            PERMANENT_COLOR,
        ),
        "permanent_dipoles",
    )
    if induced is not None:
        cmd.load_cgo(
            _dipole_cgo(system, induced, induced_scale, INDUCED_COLOR),
            "induced_dipoles",
        )


def _annotate_frame(
    path: Path,
    system: DimerSystem,
    stage: str,
    show_title: bool,
    show_stage: bool,
    transparent_background: bool,
) -> None:
    """Add an optional header and a centered color key to a PyMOL render."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    # This is 1.8 times the original width/35 annotation size.
    font_size = max(29, round(1.8 * image.width / 35))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/gnu-free/FreeSansBold.otf", font_size
        )
        key_font = ImageFont.truetype(
            "/usr/share/fonts/gnu-free/FreeSans.otf", max(23, 2 * font_size // 3)
        )
    except OSError:
        font = ImageFont.load_default()
        key_font = font

    margin = max(12, image.width // 50)
    header_parts = []
    if show_title:
        header_parts.append(system.display_name)
    if show_stage:
        header_parts.append(stage)

    if header_parts:
        header = "\n".join(header_parts)
        line_spacing = max(4, font_size // 10)
        unpositioned_box = draw.multiline_textbbox(
            (0, 0), header, font=font, spacing=line_spacing, align="center"
        )
        header_width = unpositioned_box[2] - unpositioned_box[0]
        header_x = (image.width - header_width) / 2 - unpositioned_box[0]
        header_position = (header_x, margin)
        header_box = draw.multiline_textbbox(
            header_position,
            header,
            font=font,
            spacing=line_spacing,
            align="center",
        )
        key_y = header_box[3] + max(12, font_size // 3)
    else:
        key_box = draw.textbbox((0, 0), "permanent", font=key_font)
        key_y = margin + (key_box[3] - key_box[1]) / 2

    background_fill = (255, 255, 255, 0 if transparent_background else 255)
    draw.rectangle(
        (0, 0, image.width, key_y + font_size // 2),
        fill=background_fill,
    )
    if header_parts:
        draw.multiline_text(
            header_position,
            header,
            fill=(20, 20, 20),
            font=font,
            spacing=line_spacing,
            align="center",
        )
    line_width = max(30, image.width // 25)
    line_height = max(4, font_size // 6)
    permanent_rgb = tuple(round(255 * value) for value in PERMANENT_COLOR)
    induced_rgb = tuple(round(255 * value) for value in INDUCED_COLOR)
    label_gap = max(8, font_size // 8)
    item_gap = max(30, font_size // 2)
    permanent_box = draw.textbbox((0, 0), "permanent", font=key_font)
    induced_box = draw.textbbox((0, 0), "induced", font=key_font)
    permanent_width = line_width + label_gap + permanent_box[2]
    induced_width = line_width + label_gap + induced_box[2]
    legend_width = permanent_width + item_gap + induced_width
    legend_x = (image.width - legend_width) / 2

    def draw_legend_item(
        x_position: float,
        label: str,
        label_box: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        draw.line(
            (x_position, key_y, x_position + line_width, key_y),
            fill=color,
            width=line_height,
        )
        label_height = label_box[3] - label_box[1]
        label_y = key_y - label_height / 2 - label_box[1]
        draw.text(
            (x_position + line_width + label_gap, label_y),
            label,
            fill=(20, 20, 20),
            font=key_font,
        )

    draw_legend_item(legend_x, "permanent", permanent_box, permanent_rgb)
    induced_x = legend_x + permanent_width + item_gap
    draw_legend_item(induced_x, "induced", induced_box, induced_rgb)
    image.save(path)
    image.close()


def render_frames(
    system: DimerSystem,
    output_dir: Path,
    direct: np.ndarray,
    mutual_history: list[np.ndarray],
    permanent_scale: float,
    induced_scale: float,
    rotation_degrees: float,
    show_title: bool,
    show_stage: bool,
    transparent_background: bool,
    width: int,
    height: int,
) -> list[Path]:
    """Render permanent, direct, and mutual-iteration PNGs with PyMOL."""
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-cq"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", int(not transparent_background))
    cmd.set("orthoscopic", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("specular", 0.25)
    cmd.read_pdbstr(_pdb_string(system), "dimer")
    cmd.hide("everything", "dimer")
    cmd.show("sticks", "dimer")
    cmd.show("spheres", "dimer")
    cmd.set("sphere_scale", 0.22, "dimer")
    cmd.set("stick_radius", 0.10, "dimer")
    cmd.color("gray40", "dimer and elem C")
    cmd.color("blue", "dimer and elem N")
    cmd.color("red", "dimer and elem O")
    cmd.color("gray90", "dimer and elem H")
    cmd.load_cgo(
        _dipole_cgo(
            system,
            system.permanent_dipoles,
            permanent_scale,
            PERMANENT_COLOR,
        ),
        "permanent_dipoles",
    )
    cmd.orient("dimer")
    cmd.zoom("dimer", buffer=-0.6)
    cmd.turn("z", rotation_degrees)
    # Shift by the number of visible heading lines, avoiding excess space when
    # only --show-stage or --show-title is enabled.
    heading_lines = int(show_title) + int(show_stage)
    if heading_lines > 1:
        cmd.move("y", -0.55 * (heading_lines - 1))
    view = cmd.get_view()

    rendered: list[Path] = []
    frame_data: list[tuple[str, np.ndarray | None, Path]] = [
        ("1. Permanent Dipoles", None, output_dir / "stage_1_permanent.png"),
        ("2. Direct Induced Dipoles", direct, output_dir / "stage_2_direct.png"),
    ]
    frame_data.extend(
        (
            f"3. Iteration {index}",
            dipoles,
            output_dir / f"stage_3_mutual_{index:03d}.png",
        )
        for index, dipoles in enumerate(mutual_history, start=1)
    )

    for title, induced, path in frame_data:
        _set_stage(system, induced, permanent_scale, induced_scale)
        cmd.set_view(view)
        cmd.png(str(path), width=width, height=height, dpi=200, ray=1, quiet=1)
        _annotate_frame(
            path,
            system,
            title,
            show_title,
            show_stage,
            transparent_background,
        )
        rendered.append(path)
        print(f"Rendered {path}")

    cmd.save(str(output_dir / "point_induced_dipoles.pse"))
    cmd.delete("all")
    return rendered


def discover_frames(output_dir: Path) -> list[Path]:
    """Find existing stage frames in animation order."""
    permanent = output_dir / "stage_1_permanent.png"
    direct = output_dir / "stage_2_direct.png"
    mutual = sorted(output_dir.glob("stage_3_mutual_*.png"))
    missing = [path for path in (permanent, direct) if not path.exists()]
    if missing or not mutual:
        missing_text = ", ".join(str(path) for path in missing) or "mutual frames"
        raise FileNotFoundError(
            f"Cannot use --gif-only; missing {missing_text}. "
            "Run once without --gif-only to render the PNGs."
        )
    return [permanent, direct, *mutual]


def assemble_gif(
    frame_paths: list[Path],
    gif_path: Path,
    permanent_pause: float,
    direct_pause: float,
    mutual_duration: float,
    transparent_background: bool,
) -> None:
    """Assemble frames while assigning exact stage durations."""
    from PIL import Image

    if len(frame_paths) < 3:
        raise ValueError("Expected permanent, direct, and at least one mutual frame")
    mutual_frames = len(frame_paths) - 2
    mutual_ticks = max(mutual_frames, round(100.0 * mutual_duration))
    base_ticks, extra_ticks = divmod(mutual_ticks, mutual_frames)
    mutual_durations = [
        10 * (base_ticks + (index < extra_ticks)) for index in range(mutual_frames)
    ]
    durations = [
        max(10, 10 * round(100.0 * permanent_pause)),
        max(10, 10 * round(100.0 * direct_pause)),
        *mutual_durations,
    ]

    if transparent_background:
        images = []
        for path in frame_paths:
            rgba = Image.open(path).convert("RGBA")
            alpha = rgba.getchannel("A")
            frame = rgba.convert("RGB").quantize(colors=255)
            transparent_mask = alpha.point(lambda value: 255 if value < 128 else 0)
            frame.paste(255, mask=transparent_mask)
            rgba.close()
            images.append(frame)
    else:
        images = [Image.open(path).convert("RGB") for path in frame_paths]

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    save_options = {
        "save_all": True,
        "append_images": images[1:],
        "duration": durations,
        "loop": 0,
        "disposal": 2,
        "optimize": False,
    }
    if transparent_background:
        save_options["transparency"] = 255
    images[0].save(gif_path, **save_options)
    for image in images:
        image.close()
    print(f"Wrote {gif_path} ({sum(durations) / 1000.0:.2f} s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("all", *SYSTEMS),
        default="all",
        help="System to render; default renders both into separate directories",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualize_ipd"))
    parser.add_argument("--gif-name", default="point_induced_dipoles.gif")
    parser.add_argument(
        "--gif-only", action="store_true", help="Reuse existing PNG frames"
    )
    parser.add_argument(
        "--transparent-background",
        "--transparent",
        dest="transparent_background",
        action="store_true",
        help="Render PNGs and the GIF with a transparent background",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Show the system title above the legend (default: off)",
    )
    parser.add_argument(
        "--show-stage",
        action="store_true",
        help="Show the stage or iteration above the legend (default: off)",
    )
    parser.add_argument("--permanent-pause", type=float, default=3.0)
    parser.add_argument("--direct-pause", type=float, default=3.0)
    parser.add_argument("--mutual-duration", type=float, default=5.0)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="SCF convergence threshold (default: %(default)g)",
    )
    parser.add_argument(
        "--permanent-scale",
        type=float,
        default=None,
        help="Override the system's permanent-dipole scale",
    )
    parser.add_argument(
        "--dipole-scale",
        type=float,
        default=None,
        help="Override the system's induced-dipole scale",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=None,
        help="Override clockwise image-plane rotation in degrees",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    for name in (
        "permanent_pause",
        "direct_pause",
        "mutual_duration",
        "convergence_threshold",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("permanent_scale", "dipole_scale"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    selected = SYSTEMS.values() if args.system == "all" else (SYSTEMS[args.system],)

    for system in selected:
        system_output = args.output_dir / system.key
        permanent_scale = args.permanent_scale or system.permanent_scale
        induced_scale = args.dipole_scale or system.induced_scale
        rotation = (
            args.rotation if args.rotation is not None else system.rotation_degrees
        )
        if args.gif_only:
            frames = discover_frames(system_output)
        else:
            direct, mutual_history = calculate_induced_dipoles(
                system,
                convergence_threshold=args.convergence_threshold,
            )
            final_delta = np.linalg.norm(mutual_history[-1] - mutual_history[-2])
            print(
                f"{system.display_name}: {len(mutual_history)} mutual iterations; "
                f"final SCF change = {final_delta:.3e}"
            )
            frames = render_frames(
                system,
                system_output,
                direct,
                mutual_history,
                permanent_scale,
                induced_scale,
                rotation,
                args.show_title,
                args.show_stage,
                args.transparent_background,
                args.width,
                args.height,
            )
        assemble_gif(
            frames,
            system_output / args.gif_name,
            args.permanent_pause,
            args.direct_pause,
            args.mutual_duration,
            args.transparent_background,
        )


if __name__ == "__main__":
    main()
