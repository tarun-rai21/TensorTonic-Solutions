import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    mean = np.mean(x, axis = -1, keepdims=True)
    variance = np.var(x, axis = -1, keepdims=True)    
    x_normalized = (x - mean)/np.sqrt(variance + eps)
    return gamma * x_normalized + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, L, d_model = Q.shape
    d_k = d_model // num_heads
    
    # 1. Linear projections
    Q = Q @ W_q   # (B, L, d_model)
    K = K @ W_k
    V = V @ W_v
    
    # 2. Reshape into heads
    Q = Q.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)  # (B, h, L, d_k)
    K = K.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Scaled dot-product attention
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (B, h, L, L)
    attn = softmax(scores, axis=-1)
    
    # 4. Apply attention to V
    out = attn @ V  # (B, h, L, d_k)
    
    # 5. Merge heads
    out = out.transpose(0, 2, 1, 3).reshape(B, L, d_model)  # (B, L, d_model)
    
    # 6. Final projection
    out = out @ W_o  # (B, L, d_model)
    
    return out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    layer1_out = layer_norm(mha_out + x, gamma1, beta1, eps = 1e-6)
    fnn_out = feed_forward(layer1_out, W1, b1, W2, b2)
    layer2_out = layer_norm(layer1_out + fnn_out, gamma2, beta2, eps = 1e-6)

    return layer2_out

    