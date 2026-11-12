def build_radial_cdf(rd, rmax, n_panels):
    check_positive(rd, "rd")
    check_positive(rmax, "rmax")
    check_integer(n_panels, "n_panels")
    radius_grid = np.linspace(0.0, rmax, n_panels + 1)
    weight = radius_grid * np.exp(-radius_grid / rd)
    cumulative = cumulative_trapezoid(radius_grid, weight)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ValueError("The normalization must be positive and finite.")
    cdf = cumulative / cumulative[-1]
    if np.any(np.diff(cdf) <= 0.0):
        raise ValueError("The inverse CDF needs strictly increasing values; revise the grid/range.")
    return radius_grid, cdf
