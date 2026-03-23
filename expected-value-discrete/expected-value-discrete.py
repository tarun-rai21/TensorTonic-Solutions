import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    
    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")
    
    return float(np.sum(x * p))