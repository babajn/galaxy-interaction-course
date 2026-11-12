panel_counts = [20, 40, 80, 160, 320]
mass_rows = []
for count in panel_counts:
    r = np.linspace(0.0, r_tab, count + 1)
    shell = 4.0 * np.pi * r**2 * density_plummer(r, mass, a)
    calculated = cumulative_trapezoid(r, shell)
    expected = mass_plummer_exact(r, mass, a)
    error = np.max(np.abs(calculated - expected)) / mass
    mass_rows.append([count, r_tab / count, error])
mass_convergence = np.asarray(mass_rows)
mass_orders = np.log2(mass_convergence[:-1, 2] / mass_convergence[1:, 2])
