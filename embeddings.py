import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    
class RoPE(nn.Module):
    
    def __init__(self, head_size: int, max_seq_len: int, base: int=10000) -> None:
        super().__init__()
        
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.base = base
        
        thetas = torch.tensor(
            [1 / base ** (2 * i / head_size) for i in range(head_size // 2)]
        )
        
        indices = torch.arange(max_seq_len)
        freqs = indices.float().view(-1, 1) @ thetas.view(1, -1)
        
        sines = torch.sin(freqs)
        cosines = torch.cos(freqs)

        self.register_buffer('sines', sines)
        self.register_buffer('cosines', cosines)
        
        
    def forward(self, x: torch.Tensor, start_pos: int=0) -> torch.Tensor:
        seq_len = x.shape[2]
        
        sines = self.sines[start_pos:start_pos + seq_len].unsqueeze(0) # type: ignore
        cosines = self.cosines[start_pos:start_pos + seq_len].unsqueeze(0) # type: ignore
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        new_x_even = x_even * cosines - x_odd * sines
        new_x_odd = x_odd * cosines + x_even * sines
        
        x = torch.stack([new_x_even, new_x_odd], dim=-1)
        x = x.flatten(-2)
        
        return x
    