import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Step 1: Create position indices (shape: seq_length x 1)
    pos = np.arange(seq_length).reshape(-1, 1)
    
    # Step 2: Create dimension indices (shape: 1 x d_model)
    i = np.arange(d_model).reshape(1, -1)
    
    # Step 3: Compute the denominator term (frequencies)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    
    # Step 4: Compute angle matrix
    angle = pos * angle_rates   # broadcasting happens here
    
    # Step 5: Initialize output matrix
    pe = np.zeros((seq_length, d_model))
    
    # Step 6: Apply sin to even indices
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    
    # Step 7: Apply cos to odd indices
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    
    return pe