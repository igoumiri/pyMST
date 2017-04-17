# Classic alpha model for cylindrical equilibrium

def lam(x, lam0, alpha=4.0):
    """Return classic alpha model lambda value(s) for input value(s)."""
    return lam0 * ( 1.0 - x**alpha )

