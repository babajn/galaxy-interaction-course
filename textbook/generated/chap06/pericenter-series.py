for d_peri in [5.0, 12.0]:
    config = dict(base_config, d_peri=d_peri)
    runs[f"P{d_peri:g}"] = run_case(config)
