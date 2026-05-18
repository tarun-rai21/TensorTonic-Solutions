import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y, dtype=float)

    # Empty node
    if len(y) == 0:
        return 0.0

    # Count class frequencies
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    probs = counts / len(y)

    # Remove zeros for numerical stability
    probs = probs[probs > 0]

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)