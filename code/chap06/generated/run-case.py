def run_case(config, n_steps=12000, save_every=40):
    initial, origin, radii = make_initial_state(**config)
    args = (config["masses"], config["scales"], config["b"])
    times, history, ct, ch = integrate_leapfrog(acceleration_system, initial, t_end, n_steps, *args, save_every)
    j, t_peri, d_min, tau_used = aligned_index(times, ct, ch, tau_target)
    fractions = outer_fractions(history, origin, threshold)
    return dict(config=config, initial=initial, origin=origin, radii=radii, times=times,
                history=history, center_times=ct, center_history=ch, fractions=fractions,
                index=j, t_peri=t_peri, d_min=d_min, tau_used=tau_used,
                errors=center_errors(ct, ch, config["masses"], config["b"]))
