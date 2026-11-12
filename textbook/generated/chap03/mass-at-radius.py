def enclosed_mass_trapezoid(radius, mass, a, n_panels):
    """[0, radius]をn_panels区間で積分します。"""
    check_panels(n_panels)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative.")
    if radius == 0.0:
        return 0.0
    r = np.linspace(0.0, radius, n_panels + 1)
    shell = 4.0 * np.pi * r**2 * density_plummer(r, mass, a)
    return cumulative_trapezoid(r, shell)[-1]


def half_mass_residual(radius, n_panels=n_mass):
    return enclosed_mass_trapezoid(radius, mass, a, n_panels) / mass - 0.5
