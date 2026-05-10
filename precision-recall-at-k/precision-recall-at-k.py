import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """

    recommended = recommended[:k]

    recommended_set = set(recommended)
    relevant_set = set(relevant)

    intersection_count = len(recommended_set & relevant_set)

    precision_k = intersection_count / k
    recall_k = intersection_count / len(relevant_set)

    return [float(precision_k), float(recall_k)]