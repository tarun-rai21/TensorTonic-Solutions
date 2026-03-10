import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """

    result = []
    for element in x:

        if element>0:
            result.append(element)

        else:
            result.append(alpha*(np.exp(element)-1))

    return result