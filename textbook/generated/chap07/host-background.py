def host_density(radius, m_host, a_host):
    return 3.0 * m_host / (4.0 * np.pi * a_host**3) * (1.0 + (radius / a_host)**2)**(-2.5)


def host_sigma(radius, m_host, a_host):
    return np.sqrt(G * m_host / (6.0 * np.sqrt(radius**2 + a_host**2)))
