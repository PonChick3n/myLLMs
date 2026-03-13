import dill
from typing import Self


class BPE:

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}
    
    def fit(self, text: str) -> None:
        unique_tokens = sorted(list(set(text)))
        tokens = list(text)
        
        while len(unique_tokens) < self.vocab_size:
            
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            if not pair_counts:
                break
            
            most_frequent_pair = max(pair_counts, key=pair_counts.get) # type: ignore
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            unique_tokens.append(new_token)
            
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == most_frequent_pair[0] and tokens[i + 1] == most_frequent_pair[1]:
                    tokens[i] = new_token
                    del tokens[i + 1]
                else:
                    i += 1

        for idx, token in enumerate(unique_tokens[:self.vocab_size]):
            self.id2token[idx] = token
            self.token2id[token] = idx
     
    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        sorted_tokens = sorted(self.token2id.keys(), key=len, reverse=True)
        i = 0
        
        while i < len(tokens):
            for token in sorted_tokens:
                if len(token) > 1 and i + len(token) <= len(tokens):
                    if ''.join(tokens[i:i + len(token)]) == token:
                        tokens[i] = token
                        del tokens[i+1:i + len(token)]
                        break
            i += 1
    
        encoded = [self.token2id[token] for token in tokens]
        return encoded
    
    def decode(self, token_ids: list[int]) -> str:
        decoded = ''.join([self.id2token[token_id] for token_id in token_ids])
        return decoded
    
    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f'Объект сохранён в {filename}')
    
    @classmethod
    def load(cls, filename: str) -> Self:
        with open(filename, 'rb') as f:
            obj = dill.load(f)
                
        print(f"Объект загружен из {filename}")
        return obj
        