x1, y1 = state0[0, :2]
x2, y2 = state0[1, :2]
dx, dy = x2 - x1, y2 - y1
sep2 = dx**2 + dy**2
ax1, ay1 = G * masses[1] * dx / sep2**1.5, G * masses[1] * dy / sep2**1.5
ax2, ay2 = -G * masses[0] * dx / sep2**1.5, -G * masses[0] * dy / sep2**1.5
