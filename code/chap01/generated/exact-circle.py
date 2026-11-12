def exact_circle(times, r0, mu):
    omega = np.sqrt(mu / r0**3)
    phase = omega * np.asarray(times)
    pos_exact = np.column_stack((r0 * np.cos(phase), r0 * np.sin(phase)))
    vel_exact = np.column_stack((-r0 * omega * np.sin(phase), r0 * omega * np.cos(phase)))
    return pos_exact, vel_exact

pos_exact, vel_exact = exact_circle(times, r0, mu)
pos_history = history[:, 0, :2]  # 第1軸は出力時刻です。
vel_history = history[:, 0, 2:]
