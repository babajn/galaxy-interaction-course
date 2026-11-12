def disk_cdf_exact(radius, rd, rmax):
    """切断指数円盤のCDF。clipはCDFの定義域の延長で、粒子位置は切りません。"""
    radius = np.asarray(radius, dtype=float)
    x = np.clip(radius, 0.0, rmax) / rd
    x_max = rmax / rd
    # C(x) = 1 - (1+x) exp(-x) と同じ式です。
    numerator = -np.expm1(-x) - x * np.exp(-x)
    denominator = -np.expm1(-x_max) - x_max * np.exp(-x_max)
    return numerator / denominator
