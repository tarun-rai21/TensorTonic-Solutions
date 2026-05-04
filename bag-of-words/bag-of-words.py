import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Step 1: word -> index mapping
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    # Step 2: initialize result vector
    result = np.zeros(len(vocab), dtype=int)
    
    # Step 3: count occurrences
    for token in tokens:
        if token in vocab_index:
            result[vocab_index[token]] += 1
    
    return result