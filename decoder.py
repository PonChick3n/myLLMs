import torch
import torch.nn as nn
import math
from embeddings import RoPE


HeadCache = tuple[torch.Tensor, torch.Tensor]


class MultiQueryAttention(nn.Module):

    def __init__(self, num_q_heads: int, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE, dropout: float=0.1) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.rope = rope

        self.W_q = nn.Linear(emb_size, num_q_heads * head_size)
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask)

        self.layer = nn.Linear(num_q_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache: bool=True,
                cache: HeadCache | None=None) -> tuple[torch.Tensor, HeadCache | None]:
        batch_size, seq_len, _ = x.shape

        query = self.W_q(x).view(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        key = self.W_k(x).view(batch_size, seq_len, 1, self.head_size).transpose(1, 2)
        value = self.W_v(x).view(batch_size, seq_len, 1, self.head_size).transpose(1, 2)

        if cache is not None:
            start_pos = cache[0].shape[2]
        else:
            start_pos = 0

        if self.rope is not None:
            query = self.rope(query, start_pos)
            key = self.rope(key, start_pos)

        if cache is not None:
            key = torch.cat([cache[0], key], dim=2)
            value = torch.cat([cache[1], value], dim=2)

        attention = query @ key.transpose(-2, -1)
        attention = attention / math.sqrt(self.head_size)

        if cache is None:
            k_len = key.size(-2)
            q_len = query.size(-2)

            if k_len <= self.max_seq_len:
                mask = self.mask[:q_len, :k_len] # type: ignore
            else:
                mask = torch.tril(torch.ones(q_len, k_len, device=x.device))

            while mask.dim() < attention.dim():
                mask = mask.unsqueeze(0)

            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        out = (attention @ value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_q_heads * self.head_size)
        out = self.layer(out)
        out = self.dropout(out)

        if use_cache:
            return out, (key, value)
        
        return out, None
    

class GeGLU(nn.Module):
    
    def __init__(self, emb_size: int, dropout: float=0.1):
        super().__init__()
        
        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.gate(x)
        gated = self.gelu(gated)
        
        x = self.up(x)
        x = x * gated
        
        x = self.down(x)
        x = self.dropout(x)
        
        return x
         
    
class Decoder(nn.Module):
     
    def __init__(self, num_q_heads: int, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE, dropout: float=0.1) -> None:
        super().__init__()
        
        self.grouped = MultiQueryAttention(num_q_heads, emb_size, head_size, max_seq_len, rope, dropout)
        self.geglu = GeGLU(emb_size, dropout)
        self.norm1 = nn.RMSNorm(emb_size)
        self.norm2 = nn.RMSNorm(emb_size)
    
    def forward(self, x: torch.Tensor, use_cache: bool=True,
                cache: HeadCache | None=None) -> tuple[torch.Tensor, HeadCache | None]:
        residual = x
        x = self.norm1(x)
        x, new_cache = self.grouped(x, use_cache, cache)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.geglu(x)
        x = x + residual

        if use_cache:
            return x, new_cache
        
        return x, None
    