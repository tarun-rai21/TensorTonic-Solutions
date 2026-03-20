import numpy as np

def softmax(x):
    result = []
    x = np.asarray(x, dtype=float)
    
    if np.ndim(x) > 1:
        for j in range(x.shape[0]):
            row = x[j]
            max_val = np.max(row)   # stability
            
            total = 0
            for i in range(x.shape[1]):
                total += np.exp(row[i] - max_val)
            
            row_result = []
            for k in range(x.shape[1]):
                val = np.exp(row[k] - max_val) / total
                row_result.append(val)
            
            result.append(row_result)

    else:
        max_val = np.max(x)
        
        total = 0
        for i in range(len(x)):
            total += np.exp(x[i] - max_val)
        
        for k in range(len(x)):
            val = np.exp(x[k] - max_val) / total
            result.append(val)

    return result