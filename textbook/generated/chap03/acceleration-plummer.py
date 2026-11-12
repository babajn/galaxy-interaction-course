def acceleration_plummer(pos, mass, a):
    """原点に固定したPlummer球。1粒子の位置・加速度は(1, 2)です。"""
    x, y = pos[0]
    r2 = x**2 + y**2
    ax = -G * mass * x / (r2 + a**2)**1.5
    ay = -G * mass * y / (r2 + a**2)**1.5
    return np.array([[ax, ay]])


def circular_speed_plummer(radius, mass, a):
    radius = np.asarray(radius, dtype=float)
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)
