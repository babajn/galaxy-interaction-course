def linear_interp_one(x, xp, fp):
    """一つの点を補間します。本文 eq:ch4-linear-interpolation。"""
    xp, fp = check_grid(xp, fp)
    if not np.isfinite(x) or x < xp[0] or x > xp[-1]:
        raise ValueError("Interpolate only within the finite table range.")
    if x == xp[-1]:
        return float(fp[-1])
    for j in range(len(xp) - 1):
        if xp[j] <= x <= xp[j + 1]:
            fraction = (x - xp[j]) / (xp[j + 1] - xp[j])
            return (1.0 - fraction) * fp[j] + fraction * fp[j + 1]
    raise RuntimeError("No interpolation interval was found.")
