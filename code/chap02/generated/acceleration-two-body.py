def acceleration_two_body(pos, masses):
    """2天体の同じ時刻の位置から両方の加速度を返します。"""
    dr = pos[1] - pos[0]
    sep2 = np.dot(dr, dr)
    if not np.isfinite(sep2) or sep2 <= 0.0:
        raise ValueError("Point-mass separation must be finite and nonzero.")
    common = G * dr / sep2**1.5
    acc1 = masses[1] * common
    acc2 = -masses[0] * common
    return np.array([acc1, acc2])
