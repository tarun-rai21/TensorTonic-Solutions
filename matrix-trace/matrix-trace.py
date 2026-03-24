import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A, dtype=float)
    trace = 0;
    for i in range(A.shape[0]):
        trace+=A[i][i]
    #return np.trace(A) ----> alternate solution
    return trace