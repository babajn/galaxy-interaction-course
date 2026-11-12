def plot_snapshot(state, origin, time, limit=90.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.1), layout="constrained")
    for k, (color, marker) in enumerate([(BLUE, "o"), (VERMILLION, "^")]):
        points = state[2 + np.flatnonzero(origin == k), :2]
        ax.scatter(points[:, 0], points[:, 1], s=4, color=color, marker=marker,
                   alpha=0.75, linewidths=0, label=f"Origin: galaxy {k+1}")
    ax.plot(state[0, 0], state[0, 1], "x", color="black", ms=7, mew=1.5, label="Center 1")
    ax.plot(state[1, 0], state[1, 1], "+", color="black", ms=9, mew=1.5, label="Center 2")
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title=f"t = {time:.2f} Gyr")
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.6)
    return fig
