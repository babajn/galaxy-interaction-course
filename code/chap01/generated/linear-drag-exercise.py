def rhs_linear_drag(t, state, mu):
    """発展演習：速度比例抵抗。Chandrasekhar摩擦ではありません。"""
    derivative = rhs_kepler(t, state, mu)
    derivative[:, 2:] -= gamma * state[:, 2:]
    return derivative
