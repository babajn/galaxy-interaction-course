def rhs_kepler(t, state, mu):
    """状態[x, y, vx, vy]の右辺。本文 eq:ch1-state-ode。"""
    pos, vel = state[:, :2], state[:, 2:]
    acc = acceleration_kepler(pos, mu)
    return np.column_stack((vel, acc))
