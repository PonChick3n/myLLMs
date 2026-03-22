import torch
import torch.nn as nn
import math
from embeddings import RoPE


HeadCache = tuple[torch.Tensor, torch.Tensor]


class GroupedQueryAttention(nn.Module):

    def __init__(self, num_q_heads: int, num_kv_heads: int, emb_size: int, head_size: int, max_seq_len: int,
                rope: RoPE, window_size: int, dropout: float=0.1) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.rope = rope
        self.window_size = window_size

        self.W_q = nn.Linear(emb_size, num_q_heads * head_size)
        self.W_k = nn.Linear(emb_size, num_kv_heads * head_size)
        self.W_v = nn.Linear(emb_size, num_kv_heads * head_size)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=0)
        mask = mask * torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=-window_size)
        self.register_buffer('mask', mask)

        self.layer = nn.Linear(num_q_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache: bool=True,
                cache: HeadCache | None=None) -> tuple[torch.Tensor, HeadCache | None]:
        batch_size, seq_len, _ = x.shape

        query = self.W_q(x).view(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        key = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        value = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)

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

        if use_cache:
            cache_key = key[:, :, -self.window_size:, :].clone()
            cache_value = value[:, :, -self.window_size:, :].clone()
            new_cache = (cache_key, cache_value)
        else:
            new_cache = None

        repeat_factor = self.num_q_heads // self.num_kv_heads

        B, H, S, D = key.shape
        key = key.unsqueeze(2).expand(B, H, repeat_factor, S, D).contiguous().view(B, H * repeat_factor, S, D)
        value = value.unsqueeze(2).expand(B, H, repeat_factor, S, D).contiguous().view(B, H * repeat_factor, S, D)

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

        return out, new_cache


class SwiGLU(nn.Module):
    
    def __init__(self, emb_size: int, dropout: float=0.1) -> None:
        super().__init__()
        
        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.gate(x)
        gated = self.silu(gated)
        
        x = self.up(x)
        x = x * gated
        
        x = self.down(x)
        x = self.dropout(x)
        
        return x
    

class MoE(nn.Module):
    
    def __init__(self, emb_size: int, num_experts: int, top_k_experts: int, dropout: float=0.1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.router = nn.Linear(emb_size, num_experts)
        self.experts = nn.ModuleList([SwiGLU(emb_size, dropout) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        values, indices = torch.topk(logits, self.top_k_experts, dim=-1)
        probs = torch.softmax(values, dim=-1)
        out = torch.zeros_like(x)

        for i in range(self.num_experts):
            mask = (indices == i)
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            x_i = x[token_mask]
            expert_out = self.experts[i](x_i)
            prob_i = probs[mask]
            prob_i = prob_i.unsqueeze(-1)
            expert_out = expert_out * prob_i
            out[token_mask] += expert_out

        out = self.dropout(out)
        return out
         
    
class Decoder(nn.Module):
     
    def __init__(self, num_q_heads: int, num_kv_heads: int, emb_size: int, head_size: int, max_seq_len: int,
                rope: RoPE, num_experts: int, top_k_experts: int, window_size: int, dropout: float=0.1) -> None:
        super().__init__()
        
        self.grouped = GroupedQueryAttention(num_q_heads, num_kv_heads, emb_size, head_size, max_seq_len, rope, window_size, dropout)
        self.moe = MoE(emb_size, num_experts, top_k_experts, dropout)
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
        x = self.moe(x)
        x = x + residual

        if use_cache:
            return x, new_cache
        
        return x, None