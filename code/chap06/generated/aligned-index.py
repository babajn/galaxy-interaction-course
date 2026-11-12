def aligned_index(times, center_times, center_history, tau_target):
    separation = np.linalg.norm(center_history[:, 1, :2] - center_history[:, 0, :2], axis=1)
    i_peri = int(np.argmin(separation))
    if i_peri == 0 or i_peri == len(center_times) - 1:
        raise ValueError("An interior pericenter is required for this comparison.")
    t_peri = center_times[i_peri]
    target = t_peri + tau_target
    if not times[0] <= target <= times[-1]:
        raise ValueError("The requested post-pericenter epoch is outside the stored range.")
    j = int(np.argmin(np.abs(times - target)))
    return j, t_peri, separation[i_peri], times[j] - t_peri
