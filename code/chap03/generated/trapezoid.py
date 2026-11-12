def cumulative_trapezoid(x, values):
    """各分点までの累積積分。本文 eq:ch3-cumulative-trapezoid。"""
    x, values = check_grid(x, values)
    integral = np.zeros_like(x)
    for j in range(len(x) - 1):
        area = 0.5 * (values[j] + values[j + 1]) * (x[j + 1] - x[j])
        integral[j + 1] = integral[j] + area
    return integral
