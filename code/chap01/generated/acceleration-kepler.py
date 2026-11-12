def acceleration_kepler(pos, mu):
    """1粒子の重力加速度。本文 eq:ch1-acceleration-components。"""
    x, y = pos[0]
    r2 = x**2 + y**2
    if r2 <= 0.0:
        raise ValueError("Point-mass gravity is undefined at r = 0.")
    ax = -mu * x / r2**1.5
    ay = -mu * y / r2**1.5
    return np.array([[ax, ay]])
