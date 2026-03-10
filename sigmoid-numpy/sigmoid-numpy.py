import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x, dtype=float) #converts scalars, lists and nested lists into numpy array
    return 1/(1+np.exp(-x))
    pass