import numpy as np

def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    values = np.asarray(values, dtype=float)

    n = len(values)

    if window_size > n or window_size <= 0:
        return []

    result = []

    # First window sum
    window_sum = np.sum(values[:window_size])
    result.append(window_sum / window_size)

    # Sliding the window
    for i in range(window_size, n):
        window_sum += values[i] - values[i - window_size]
        result.append(window_sum / window_size)

    return result