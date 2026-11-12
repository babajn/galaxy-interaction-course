base_config = dict(masses=np.array([1e11, 1e11]), scales=np.array([3.0, 3.0]), b=3.0,
                   rds=np.array([3.0, 3.0]), rmaxs=np.array([15.0, 15.0]),
                   counts=np.array([1000, 1000]), spins=np.array([1, 1]),
                   d_start=60.0, d_peri=8.0, seed=42, n_cdf=4096)
t_end, n_steps, save_every = 0.60, 12000, 40
threshold, tau_target = 30.0, 0.20  # [kpc], [Gyr]
