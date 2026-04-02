import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x-y

    sum = 0
    for item in diff:
        sum+=item**2

    return np.sqrt(sum)