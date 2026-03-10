import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """

    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)
    pass

# np.max() returns the maximum value in an array
# np.maximum() performs element-wise comparison between two arrays/values