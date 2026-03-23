def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp = 0 #number of correct predictions
    
    """
    But each mismatch contributes to BOTH FP and FN
    FP+FN=2⋅(mismatches)
    2(TP) + FN + FP = 2(TP) + 2⋅(mismatches) = 2(TP + mismatches) =       2(Total)
    """
    

    for i in range(len(y_pred)):
        if(y_pred[i]==y_true[i]):
            tp+=1
    f1_micro = tp/len(y_pred)
    return f1_micro