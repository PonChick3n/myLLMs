import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    
class PositionalEmbeddings(nn.Module):
    
    def __init__(self, max_seq_len: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, emb_size)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        return self.embedding(torch.arange(seq_len))
    