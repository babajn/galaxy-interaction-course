def chandrasekhar_acceleration(pos, vel, params):
    radius, speed = np.linalg.norm(pos), np.linalg.norm(vel)
    rho = host_density(radius, params["m_host"], params["a_host"])
    sigma = host_sigma(radius, params["m_host"], params["a_host"])
    scale_v = np.sqrt(2.0) * sigma
    x = speed / scale_v
    gamma = 4.0 * np.pi * G**2 * params["m_sat"] * rho * params["ln_lambda"]
    gamma *= maxwell_factor_over_x3(x) / scale_v**3
    return -gamma * vel
