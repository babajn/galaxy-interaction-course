orbit_rows = []
for count in [750, 1500, 3000, 6000]:
    t_check, h_check = integrate_leapfrog(acceleration_plummer, state0, t_end, count, mass, a, count // 150)
    errors = disk_error_metrics(t_check, h_check, state0, mass, a, rd, spin)
    orbit_rows.append([count, t_end / count, *errors])
orbit_convergence = np.asarray(orbit_rows)
orbit_orders = np.log2(orbit_convergence[:-1, 2] / orbit_convergence[1:, 2])
