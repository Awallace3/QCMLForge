def is_psi4_installed():
    """Check if Psi4 is installed."""
    try:
        import psi4

        return True
    except ImportError:
        return False


try:
    from . import all_polynomial_fits
    from . import estimate_timings
except ImportError as e:
    print(f"Error importing modules: {e}")
