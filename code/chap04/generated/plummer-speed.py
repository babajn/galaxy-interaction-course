def circular_speed_plummer(radius, mass, a):
    radius = np.asarray(radius, dtype=float)
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)


def potential_plummer(radius, mass, a):
    return -G * mass / np.sqrt(np.asarray(radius)**2 + a**2)
