fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
ax.plot(pos_exact[:, 0], pos_exact[:, 1], color=GRAY, ls="--", lw=2, label="Exact circle")
ax.plot(pos_history[:, 0], pos_history[:, 1], color=BLUE, lw=1.3, label="Leapfrog (KDK)")
ax.scatter([0.0], [0.0], marker="*", s=90, color="black", label="Central mass")
sample_ids = np.rint(np.arange(4) * n_steps / 4).astype(int)
for n in sample_ids:
    ax.plot(*pos_history[n], marker="s", color=BLUE, ms=5)
    label = f"{times[n] / period:.2f} T"
    ax.annotate(label, pos_history[n], xytext=(7, 7), textcoords="offset points")
ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-13, 13), ylim=(-13, 13))
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=9)
save_figure(fig, "ch01_orbit")
