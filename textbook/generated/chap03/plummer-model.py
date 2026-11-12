def density_plummer(radius, mass, a):
    """3次元の質量密度 [Msun/kpc^3]。本文 eq:ch3-density。"""
    radius = np.asarray(radius, dtype=float)
    return 3.0 * mass / (4.0 * np.pi * a**3) * (1.0 + (radius / a)**2)**(-2.5)


def potential_plummer(radius, mass, a):
    """単位質量当たりのポテンシャル [kpc^2/Gyr^2]。"""
    return -G * mass / np.sqrt(np.asarray(radius)**2 + a**2)


def mass_plummer_exact(radius, mass, a):
    """半径radius以内の解析的な質量。本文 eq:ch3-mass-exact。"""
    radius = np.asarray(radius, dtype=float)
    return mass * radius**3 / (radius**2 + a**2)**1.5
