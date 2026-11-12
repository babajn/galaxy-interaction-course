def leapfrog_step(acceleration, pos, vel, acc, dt, mu):
    """KDKで1ステップ。accは現在位置での加速度です。"""
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, mu)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
