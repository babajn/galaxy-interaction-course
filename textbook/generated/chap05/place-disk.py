def place_disk(local_state, center_state):
    """局所の位置と速度を慣性系へ移します。入力を上書きしません。"""
    placed = local_state.copy()
    placed[:, :2] += center_state[:2]
    placed[:, 2:] += center_state[2:]
    return placed
