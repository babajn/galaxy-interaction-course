step_counts = [100, 200, 400, 800]
convergence_rows = []
for count in step_counts:
    tc, hc = integrate_leapfrog(acceleration_two_body, cases["Circular"]["initial"], t_c, count, masses)
    dr_c = hc[:, 1, :2] - hc[:, 0, :2]
    exact_c = exact_relative_circle(tc, d0, mu)
    err_c = np.max(np.linalg.norm(dr_c - exact_c, axis=1)) / d0
    te, he = integrate_leapfrog(acceleration_two_body, cases["Elliptic"]["initial"], t_ell, count, masses)
    dr_end = he[-1, 1, :2] - he[-1, 0, :2]
    err_ell = np.linalg.norm(dr_end - np.array([d0, 0.0])) / d0
    err_energy_ell = conservation_errors(te, he, masses)[0]
    convergence_rows.append([count, t_c / count, t_ell / count, err_c, err_ell, err_energy_ell])
convergence = np.array(convergence_rows)
orders_c = np.log2(convergence[:-1, 3] / convergence[1:, 3])
orders_ell = np.log2(convergence[:-1, 4] / convergence[1:, 4])
