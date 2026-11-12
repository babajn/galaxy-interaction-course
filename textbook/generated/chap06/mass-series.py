for q in [0.5, 0.1]:
    config = dict(base_config, masses=np.array([1.0e11, q * 1.0e11]))
    runs[f"Q{q:g}"] = run_case(config)
