import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        # add special tokens
        for token in special_tokens:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        vocab = set()

        for text in texts:
            vocab.update(text.lower().split())

        for word in sorted(vocab):
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)

    
    def encode(self, text: str) -> List[int]:

        tokens = text.lower().split()

        return [
            self.word_to_id.get(token, self.word_to_id[self.unk_token])
            for token in tokens
        ]
    
    
    def decode(self, ids: List[int]) -> str:

        words = [
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        ]

        return " ".join(words)