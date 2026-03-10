import numpy as np
def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)

    if len(a)==0:
        return 0 
    else:
        intersect = a&b
        union = a|b

        return len(intersect)/len(union)