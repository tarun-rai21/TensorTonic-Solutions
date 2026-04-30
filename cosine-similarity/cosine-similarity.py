import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if(a_norm == 0 or b_norm == 0):
        return 0

    return np.dot(a, b)/(a_norm * b_norm)