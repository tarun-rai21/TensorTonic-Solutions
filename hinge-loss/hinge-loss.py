import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_true = np.asarray(y_true, dtype = float)
    y_score = np.asarray(y_score, dtype = float)

    ans = []
    for i in range(len(y_true)):
        ans.append(max(0, (margin-y_score[i]*y_true[i])))

    if (reduction=="mean"):
        return np.mean(ans)
    else:
        return np.sum(ans)