def cumulative_trapezoid(x, values):
    x, values = check_grid(x, values)
    integral = np.zeros_like(x)
    for j in range(len(x) - 1):
        integral[j + 1] = integral[j] + 0.5 * (values[j] + values[j + 1]) * (x[j + 1] - x[j])
    return integral


def build_radial_cdf(rd, rmax, n_panels):
    check_positive(rd, "rd")
    check_positive(rmax, "rmax")
    check_integer(n_panels, "n_panels")
    nodes = np.linspace(0.0, rmax, n_panels + 1)
    integral = cumulative_trapezoid(nodes, nodes * np.exp(-nodes / rd))
    if not np.isfinite(integral[-1]) or integral[-1] <= 0.0:
        raise ValueError("The CDF normalization must be finite and positive.")
    cdf = integral / integral[-1]
    if np.any(np.diff(cdf) <= 0.0):
        raise ValueError("The inverse CDF needs strictly increasing values.")
    return nodes, cdf
