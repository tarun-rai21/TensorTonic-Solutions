import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimiser update step.
    Return (param_new, m_new, v_new).
    """
    # converting all parameters to float
    param = np.asarray(param, dtype = float)
    grad  = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    
    m_new = beta1*m + (1-beta1)*grad
    v_new = beta2*v + (1-beta2)*(grad**2)
    
    m_corrected = m_new/(1-beta1**t)
    v_corrected = v_new/(1-beta2**t)

    param_new = param - lr*m_corrected/(np.sqrt(v_corrected)+eps)

    return param_new, m_new, v_new