import numpy as np
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    values = np.asarray(values, dtype=float)
    
    min_val = min(values)
    max_val = max(values)

    #edge case: if all values are equal
    if (max_val == min_val):
        return [0]*len(values)

    #bin width
    w = (max_val - min_val)/num_bins
    
    bins = []
    for val in values:
        b = int((val-min_val)//w)
        bins.append(min(b, num_bins-1))

    return bins