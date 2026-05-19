import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x, dtype=float)

    freq = Counter(x)
    max_freq = max(freq.values())

    mode = min(k for k, v in freq.items() if v == max_freq)

    return (
        float(np.mean(x)),
        float(np.median(x)),
        float(mode)
    )