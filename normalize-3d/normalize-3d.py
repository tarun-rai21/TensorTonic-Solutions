import numpy as np

def normalize_3d(v):
    """
    Normalise 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype=float)
    
    # Compute magnitude
    magnitude = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Avoid division by zero
    magnitude[magnitude == 0] = 1
    
    return v / magnitude
    