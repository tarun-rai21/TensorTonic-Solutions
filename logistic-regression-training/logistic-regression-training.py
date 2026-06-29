import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for i in range(0, steps):
        p = _sigmoid(np.dot(X, w) + b)
        dl_dw = np.dot(np.transpose(X), (p - y))/X.shape[0]
        dl_db = np.mean(p - y)

        w = w - lr*dl_dw
        b = b - lr*dl_db

    return w, b