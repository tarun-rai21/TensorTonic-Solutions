import numpy as np

def angle_between_3d(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    # Handle zero vectors
    if v_norm == 0 or w_norm == 0:
        return np.nan   # or np.nan depending on problem requirement

    cos_theta = np.dot(v, w) / (v_norm * w_norm)

    # Clip for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    return theta