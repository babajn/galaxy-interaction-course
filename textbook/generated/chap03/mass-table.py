r_grid = np.linspace(0.0, r_tab, n_mass + 1)
rho_grid = density_plummer(r_grid, mass, a)
grid_shell = 4.0 * np.pi * r_grid**2 * rho_grid
mass_numeric = cumulative_trapezoid(r_grid, grid_shell)
mass_exact = mass_plummer_exact(r_grid, mass, a)
