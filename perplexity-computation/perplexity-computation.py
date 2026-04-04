import numpy as np

def perplexity(prob_distributions, actual_tokens):
    prob_distributions = np.array(prob_distributions, dtype=float)
    actual_tokens = np.array(actual_tokens, dtype=int)

    N = len(actual_tokens)
    
    log_sum = 0.0
    
    for i in range(N):
        p = prob_distributions[i][actual_tokens[i]]
        log_sum += np.log(p)
    
    cross_entropy = -log_sum / N
    
    return float(np.exp(cross_entropy))