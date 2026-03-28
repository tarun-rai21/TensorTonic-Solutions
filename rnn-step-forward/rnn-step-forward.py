import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t = np.asarray(x_t, dtype = float)
    h_prev = np.asarray(h_prev, dtype = float)

    x = np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))