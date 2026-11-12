def leapfrog_step(acceleration, pos, vel, acc, dt, masses, scales, b):
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, masses, scales, b)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
