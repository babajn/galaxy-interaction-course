def exact_circle(times, radius, mass, a):
    omega = circular_speed_plummer(radius, mass, a) / radius
    phase = omega * np.asarray(times)
    return np.column_stack((radius * np.cos(phase), radius * np.sin(phase)))


orbit_rows = []
initial_circle = cases["Circular"]["initial"]
for count in [100, 200, 400, 800]:
    t, h = integrate_leapfrog(acceleration_plummer, initial_circle, t_c, count, mass, a)
    exact = exact_circle(t, r0, mass, a)
    position_error = np.max(np.linalg.norm(h[:, 0, :2] - exact, axis=1)) / r0
    orbit_rows.append([count, t_c / count, position_error])
orbit_convergence = np.asarray(orbit_rows)
orbit_orders = np.log2(orbit_convergence[:-1, 2] / orbit_convergence[1:, 2])
