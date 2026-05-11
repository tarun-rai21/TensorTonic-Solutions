import numpy as np
def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    values = np.asarray(values, dtype = float)
    result = []

    n = len(values)
    for i in range(0, n-window_size+1):
        sum = 0
        for j in range(0, window_size):
            sum += values[i+j]
        avg = sum/window_size
        result.append(avg)

    return result