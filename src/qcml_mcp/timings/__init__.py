def is_psi4_installed():
    """Check if Psi4 is installed."""
    try:
        import psi4

        return True
    except ImportError:
        return False

from . import all_polynomial_fits
from . import estimate_timings
