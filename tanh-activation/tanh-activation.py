import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x, dtype=float) #convert the input to numpy array
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    pass