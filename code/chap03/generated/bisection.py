def bisection(func, lower, upper, xtol, max_iter=100):
    """半区間幅がxtol以下になるまで二分します。本文 eq:ch3-bisection。"""
    f_lower, f_upper = check_bisection(func, lower, upper, xtol, max_iter)
    if f_lower == 0.0 or f_upper == 0.0:
        endpoint = lower if f_lower == 0.0 else upper
        return endpoint, np.array([[0, endpoint, endpoint, endpoint, 0.0, 0.0]])
    trace = []
    for k in range(max_iter):
        half_width = 0.5 * (upper - lower)
        midpoint = lower + half_width
        f_mid = finite_value(func, midpoint)
        trace.append([k, lower, upper, midpoint, f_mid, half_width])
        if f_mid == 0.0 or half_width <= xtol:
            return midpoint, np.asarray(trace)
        if midpoint == lower or midpoint == upper:
            raise RuntimeError("The bracket cannot be reduced in floating-point arithmetic.")
        if np.signbit(f_mid) == np.signbit(f_lower):
            lower, f_lower = midpoint, f_mid
        else:
            upper, f_upper = midpoint, f_mid
    raise RuntimeError("Bisection did not reach xtol within max_iter.")
