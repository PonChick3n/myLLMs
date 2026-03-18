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
        
    def forward(self, x: torch.Tensor, start_pos: int=0) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(0)
        return self.embedding(positions)
    