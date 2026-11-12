def acceleration_plummer(pos, center, mass, a):
    """任意の中心を持つPlummer場。posは(N, 2)、centerは(2,)です。"""
    relative = pos - center
    radius2 = np.sum(relative**2, axis=1)
    return -G * mass * relative / (radius2 + a**2)[:, None]**1.5
