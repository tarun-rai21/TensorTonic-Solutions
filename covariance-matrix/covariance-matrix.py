import numpy as np

def covariance_matrix(x):
    """
    Compute covariance matrix from dataset X.
    """
    if(np.shape(x)[0]>=2 and np.ndim(x)==2):
        x = np.asarray(x, dtype = float)
        x_centered = x - np.mean(x, axis=0)
        cov_matrix = np.dot(np.transpose(x_centered), x_centered)/(np.shape(x)[0]-1)
        return cov_matrix
    
    else:
        return None
    
    pass