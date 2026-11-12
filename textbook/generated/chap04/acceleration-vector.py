def acceleration_plummer(pos, mass, a):
    """固定Plummer球から各粒子への加速度。本文 eq:ch4-acceleration。"""
    r2 = np.sum(pos**2, axis=1)
    acc = -G * mass * pos / (r2 + a**2)[:, None]**1.5
    return acc
