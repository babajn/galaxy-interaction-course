def maxwell_factor_over_x3(x):
    """[erf(x)-2x exp(-x^2)/sqrt(pi)]/x^3 の連続な低速極限を使います。"""
    if not np.isfinite(x) or x < 0.0:
        raise ValueError("x must be finite and nonnegative.")
    if x < 1.0e-3:
        return 4.0 / (3.0 * np.sqrt(np.pi)) * (1.0 - 0.6 * x**2 + 3.0 * x**4 / 14.0)
    return (erf(x) - 2.0 * x * np.exp(-x*x) / np.sqrt(np.pi)) / x**3
