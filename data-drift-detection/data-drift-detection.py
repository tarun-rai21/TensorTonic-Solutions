import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    reference_counts = np.asarray(reference_counts, dtype = float)
    production_counts = np.asarray(production_counts, dtype = float)

    # Normalize both histograms
    reference_counts = reference_counts/np.sum(reference_counts)
    production_counts = production_counts/np.sum(production_counts)

    # compute TVD
    tvd = np.sum(np.abs(reference_counts - production_counts)) / 2

    if tvd>threshold:
        drift = True
    else:
        drift = False

    result = {
        "score" : tvd,
        "drift_detected" : drift
    }

    return result