"""Frozen per-element tables for FF-1 (Z = 1..54).

Every table is a tuple indexed by atomic number (index 0 unused).  Units are
Hartree atomic units (bohr, bohr^-1, bohr^3, E_h) unless the name says
otherwise.  Shell split (proposal Eqs. 4-6):

* ``N_CORE`` -- frozen noble/d10 core: 0 (H-Be), 2 (B-Mg), 10 (Al-Zn),
  28 (Ga-Cd), 46 (In-Xe).
* ``Z_SUB`` / ``Z_VAL`` -- active electrons ``Z_act = Z - N_core`` split into
  compact subvalence and outer valence: Z_sub = 2 (Li, Be); 8 (Na, Mg, K, Ca,
  Rb, Sr); Z_act - Z_val for 3d Sc-Zn and 4d Y-Cd with Z_val = 1 for Cr, Cu,
  Nb, Mo, Ru, Rh, Pd, Ag and 2 otherwise; 0 for the 1s/p block.

Approximate tables (frozen, refined in training by bounded per-atom heads):

* ``ALPHA_FREE``/``C6_FREE``: Tkatchenko-Scheffler free-atom values.
* ``ALPHA_CAT``: polarizability of the valence-stripped closed-shell ion
  (``alpha_core + Delta alpha_sub``); Li+, Na+, K+, Mg2+, Ca2+ from Mitroy,
  Safronova & Clark (2010); Al3+ 0.265, Zn2+ 2.29, In3+ 3.29 as in the
  proposal; the rest are hydrogenic/isoelectronic estimates.
* ``ETA_CAT``: Slater-Kirkwood ``0.57 sqrt(N_shell / alpha_cat)``, calibrated
  so Na+ gives the proposal's C6 = 1.61; ``ETA_CAT = ETA_FREE`` for H and He.
* ``BETA_CORE``: 1s-Slater density exponent ``2 zeta`` of the outermost core
  subshell, zeta from Slater's screening rules (a stand-in for the
  Clementi-Raimondi value; only cross-penetration with heavy cores uses it).
* ``BETA_VAL``: valence density exponent ``1.2 * 2 sqrt(2 IP)``.
* ``R42``: Grimme D3 ``sqrt(0.5 sqrt(Z) <r^4>/<r^2>)`` (bohr).
* ``R_COV``: Cordero (2008) covalent radii (Angstrom).
"""

from __future__ import annotations

import math

MAX_Z = 54
_NONE = float("nan")

# fmt: off
IP_EV = (_NONE,
    13.598, 24.587, 5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565,
    5.139, 7.646, 5.986, 8.152, 10.487, 10.360, 12.968, 15.760, 4.341, 6.113,
    6.561, 6.828, 6.746, 6.767, 7.434, 7.902, 7.881, 7.640, 7.726, 9.394,
    5.999, 7.899, 9.789, 9.752, 11.814, 14.000, 4.177, 5.695, 6.217, 6.634,
    6.759, 7.092, 7.280, 7.361, 7.459, 8.337, 7.576, 8.994, 5.786, 7.344,
    8.608, 9.010, 10.451, 12.130)
ALPHA_FREE = (_NONE,
    4.5, 1.38, 164.2, 38.0, 21.0, 12.0, 7.4, 5.4, 3.8, 2.67,
    162.7, 71.0, 60.0, 37.0, 25.0, 19.6, 15.0, 11.1, 292.9, 160.0,
    120.0, 98.0, 84.0, 78.0, 63.0, 56.0, 50.0, 48.0, 42.0, 40.0,
    60.0, 41.0, 29.0, 25.0, 20.0, 16.8, 319.2, 199.0, 126.737, 119.97,
    101.603, 88.42, 80.08, 65.89, 56.1, 23.68, 50.6, 39.7, 70.22, 55.95,
    43.67, 37.65, 35.0, 27.3)
C6_FREE = (_NONE,
    6.5, 1.46, 1387.0, 214.0, 99.5, 46.6, 24.2, 15.6, 9.52, 6.38,
    1556.0, 627.0, 528.0, 305.0, 185.0, 134.0, 94.6, 64.3, 3897.0, 2221.0,
    1383.0, 1044.0, 832.0, 602.0, 552.0, 482.0, 408.0, 373.0, 253.0, 284.0,
    498.0, 354.0, 246.0, 210.0, 162.0, 129.6, 4691.0, 3170.0, 1968.0, 1677.0,
    1263.0, 1028.0, 1390.0, 609.0, 469.0, 157.5, 339.0, 452.0, 707.0, 587.0,
    459.0, 396.0, 385.0, 285.9)
ALPHA_CAT = (_NONE,
    0.0, 0.0, 0.1925, 0.0523, 0.0196, 0.0088, 0.0045, 0.0025, 0.0015, 0.0009,
    0.9947, 0.4814, 0.265, 0.165, 0.11, 0.075, 0.053, 0.038, 5.354, 3.161,
    7.5, 6.5, 5.5, 8.0, 4.5, 4.0, 3.6, 3.2, 5.4, 2.29,
    1.24, 0.9, 0.65, 0.5, 0.4, 0.32, 9.08, 5.81, 10.0, 9.0,
    12.0, 11.0, 7.5, 9.5, 9.0, 8.5, 9.56, 4.97, 3.29, 2.6,
    2.0, 1.6, 1.3, 1.05)
R42 = (_NONE,
    2.00734898, 1.56637132, 5.01986934, 3.85379032, 3.64446594, 3.10492822,
    2.71175247, 2.59361680, 2.38825250, 2.21522516, 6.58585536, 5.46295967,
    5.65216669, 4.88284902, 4.29727576, 4.04108902, 3.72932356, 3.44677275,
    7.97762753, 7.07623947, 6.60844053, 6.28791364, 6.07728703, 5.54643096,
    5.80491167, 5.58415602, 5.41374528, 5.28497229, 5.22592821, 5.09817141,
    6.12149689, 5.54083734, 5.06696878, 4.87005108, 4.59089647, 4.31176304,
    9.55461698, 8.67396077, 7.97210197, 7.43439917, 6.58711862, 6.19536215,
    6.01517290, 5.81623410, 5.65710424, 5.52640661, 5.44263305, 5.58285373,
    7.02081898, 6.46815523, 5.98089120, 5.81686657, 5.53321815, 5.25477007)
R_COV = (_NONE,
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40)
# fmt: on

EV2HARTREE = 1.0 / 27.211386245988


def n_core(z: int) -> int:
    """Frozen core electron count N_core(Z)."""
    if z <= 4:
        return 0
    if z <= 12:
        return 2
    if z <= 30:
        return 10
    if z <= 48:
        return 28
    return 46


_ONE_VALENCE = {24, 29, 41, 42, 44, 45, 46, 47}  # Cr Cu Nb Mo Ru Rh Pd Ag


def shell_split(z: int) -> tuple[int, int]:
    """(Z_sub, Z_val) of the active electrons Z_act = Z - N_core."""
    act = z - n_core(z)
    if z in (3, 4):
        return 2, act - 2
    if z in (11, 12, 19, 20, 37, 38):
        return 8, act - 8
    if 21 <= z <= 30 or 39 <= z <= 48:
        val = 1 if z in _ONE_VALENCE else 2
        return act - val, val
    return 0, act


def _slater_core_zeta(z: int) -> float:
    """Slater-rule zeta of the outermost core subshell (1.0 when no core)."""
    nc = n_core(z)
    if nc == 0:
        return 1.0
    if nc == 2:
        return z - 0.30
    if nc == 10:  # 2p: 2 x 0.85 (1s) + 7 x 0.35 (2s2p)
        return (z - 4.15) / 2.0
    if nc == 28:  # 3d: 18 inner x 1.0 + 9 x 0.35
        return (z - 21.15) / 3.0
    return (z - 39.15) / 3.7  # 4d, n* = 3.7


def _core_alpha(z: int) -> float:
    """Polarizability of the frozen N_core shell."""
    nc = n_core(z)
    if nc == 0:
        return 0.0
    if shell_split(z)[0] == 0:  # p block: the core IS the stripped cation
        return ALPHA_CAT[z]
    return {2: 1e-4, 10: 1e-2, 28: 0.2, 46: 1.0}[nc]


def _eta_cat(z: int) -> float:
    if z <= 2:
        return C6_FREE[z] / ALPHA_FREE[z] ** 2
    sub = shell_split(z)[0]
    shell = sub if sub > 0 else {2: 2, 10: 8, 28: 18, 46: 18}[n_core(z)]
    return 0.57 * math.sqrt(shell / ALPHA_CAT[z])


def table(name: str) -> list[float]:
    """Element table ``name`` as a list indexed by Z (index 0 is a dummy 1.0)."""
    out = [1.0]
    for z in range(1, MAX_Z + 1):
        sub, val = shell_split(z)
        a_core = _core_alpha(z)
        value = {
            "n_core": float(n_core(z)),
            "z_act": float(z - n_core(z)),
            "z_sub": float(sub),
            "z_val": float(val),
            "beta_core": 2.0 * _slater_core_zeta(z),
            "beta_val": 1.2 * 2.0 * math.sqrt(2.0 * IP_EV[z] * EV2HARTREE),
            "alpha_core": a_core,
            "d_alpha_sub": max(ALPHA_CAT[z] - a_core, 0.0) if sub > 0 else 0.0,
            "d_alpha_val": ALPHA_FREE[z] - max(ALPHA_CAT[z], a_core),
            "alpha_free": ALPHA_FREE[z],
            "eta_free": C6_FREE[z] / ALPHA_FREE[z] ** 2,
            "eta_cat": _eta_cat(z),
            "r42": R42[z],
            "r_cov": R_COV[z],
        }[name]
        out.append(float(value))
    return out


TABLE_NAMES = ("n_core", "z_act", "z_sub", "z_val", "beta_core", "beta_val",
               "alpha_core", "d_alpha_sub", "d_alpha_val", "alpha_free",
               "eta_free", "eta_cat", "r42", "r_cov")
