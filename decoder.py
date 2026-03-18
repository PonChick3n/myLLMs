import torch
import torch.nn as nn
import math


class HeadAttention(nn.Module):
    
    def __init__(self, emb_size, head_size, max_seq_len):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
    
    def forward(self, x, use_cache=True, cache=None):
        seq_len = x.shape[1]
        trimmed_mask = self.mask[:seq_len, :seq_len] # type: ignore
        
        key = self.W_k(x)
        query = self.W_q(x)
        value = self.W_v(x)
        
        if cache is not None:
            key = torch.cat([cache[0], key], dim=1)
            value = torch.cat([cache[1], value], dim=1)
              
        attention = query @ key.transpose(-2, -1)
        attention = attention / math.sqrt(self.head_size)
        
        if cache is None:
            attention = attention.masked_fill(trimmed_mask == 0, float('-inf'))
            
        attention = torch.softmax(attention, dim=-1)
        
        out = attention @ value
        
        if use_cache:
            new_cache = (key, value)
            return out, new_cache
            
        return out, None


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        self.heads = nn.ModuleList([HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)])
        self.layer = nn.Linear(head_size * num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, use_cache=True, cache=None):
        outs = []
        new_cache = []

        for i, head in enumerate(self.heads):
            head_cache = None if cache is None else cache[i]
            out, c = head(x, use_cache, head_cache)
            outs.append(out)
            new_cache.append(c)

        out = torch.cat(outs, dim=-1)
        out = self.layer(out)
        out = self.dropout(out) # type: ignore

        if use_cache:
            return out, new_cache
        
        return out, None
    
    
class FeedForward(nn.Module):
    
    def __init__(self, emb_size, dropout=0.1):
        super().__init__()
        
        self.emb_size = emb_size
        self.pipeline = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),                          
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.pipeline(x)     

    
class Decoder(nn.Module):
     
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.multihead = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.feedforward = FeedForward(emb_size, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
    
    def forward(self, x, use_cache=True, cache=None):
        residual = x
        x = self.norm1(x)
        x, new_cache = self.multihead(x, use_cache, cache)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x + residual

        if use_cache:
            return x, new_cache
        
        return x, None