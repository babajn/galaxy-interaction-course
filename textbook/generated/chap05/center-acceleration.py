def acceleration_centers(pos, masses, b):
    """共通のbを使う有効対ポテンシャルから、二つの加速度を返します。"""
    dr = pos[1] - pos[0]
    denom2 = np.dot(dr, dr) + b**2
    if not np.isfinite(denom2) or denom2 <= 0.0:
        raise ValueError("The effective force denominator must be positive and finite.")
    common = G * dr / denom2**1.5
    return np.array([masses[1] * common, -masses[0] * common])
