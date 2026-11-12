def origin_radii(history, origin):
    parent_centers = history[:, origin, :2]
    return np.linalg.norm(history[:, 2:, :2] - parent_centers, axis=2)


def outer_fractions(history, origin, threshold):
    radius = origin_radii(history, origin)
    fractions = np.empty((len(history), 2))
    for k in range(2):
        selected = origin == k
        if not np.any(selected):
            raise ValueError("Both origin groups must contain test particles.")
        fractions[:, k] = np.mean(radius[:, selected] > threshold, axis=1)
    return fractions
