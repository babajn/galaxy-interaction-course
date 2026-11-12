def exact_disk_positions(times, state0, mass, a, spin):
    """円運動の初期条件にだけ使う、同時刻の解析位置です。"""
    pos0 = state0[:, :2]
    radius2 = np.sum(pos0**2, axis=1)
    omega = np.sqrt(G * mass) / (radius2 + a**2)**0.75
    angle = spin * np.asarray(times)[:, None] * omega[None, :]
    c, s = np.cos(angle), np.sin(angle)
    x = pos0[None, :, 0] * c - pos0[None, :, 1] * s
    y = pos0[None, :, 0] * s + pos0[None, :, 1] * c
    return np.stack((x, y), axis=2)
