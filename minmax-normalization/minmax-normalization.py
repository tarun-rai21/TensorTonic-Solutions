import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype = float)

    if X.ndim == 1:
        max_val = np.max(X)
        min_val = np.min(X)
    else:
        max_val = np.max(X, axis=axis, keepdims=True)
        min_val = np.min(X, axis=axis, keepdims=True)

    denominator = np.maximum(max_val - min_val, eps) # handles division by zero

    return (X - min_val)/denominator