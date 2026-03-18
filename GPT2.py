import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from embeddings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder
from tqdm.auto import tqdm
from decoder import MultiHeadCache


class GPT2(nn.Module):
    
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int, head_size: int, num_layers: int,
                dropout: float=0.1, device: str='cpu') -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.decoders = nn.ModuleList(Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers))
        self.linear = nn.Linear(emb_size, vocab_size)
        self.norm = nn.LayerNorm(emb_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, use_cache: bool=True,
                cache: list[MultiHeadCache] | None=None) -> tuple[torch.Tensor, list[MultiHeadCache] | None]:
        token_emb = self.token_emb(x)
        
        if cache is not None:
            last_layer_cache = cache[-1]
            last_head_cache = last_layer_cache[-1]
            key_tensor = last_head_cache[0]        
            start_pos = key_tensor.shape[1] % self.max_seq_len
            pos_emb = self.pos_emb(x, start_pos)
        else:
            pos_emb = self.pos_emb(x)
            
        pos_emb = pos_emb.unsqueeze(0)
        emb = token_emb + pos_emb
        emb = self.dropout(emb)
        
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            layer_cache = cache[i] if cache is not None else None
            emb, layer_cache = decoder(emb, use_cache, layer_cache)
            new_cache.append(layer_cache)
        
        out = self.norm(emb)
        out = self.linear(out)
        
        if use_cache:
            return out, new_cache
        
        return out, None
    
    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool, top_k: int=None, top_p: float=None,
                 temperature: float=1.0, use_cache: bool=True) -> torch.Tensor:
        cache = None
        x_input = x
        for _ in range(max_new_tokens):
            logits, cache = self.forward(x_input, use_cache, cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                kth = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < kth] = float('-inf')

            if top_p is not None and top_p < 1.0:
                probs_for_p = self.softmax(logits)

                sorted_probs, sorted_indices = torch.sort(probs_for_p, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove_mask = cumulative_probs > top_p
                remove_mask[..., 0] = 0

                sorted_logits = torch.gather(logits, -1, sorted_indices)
                sorted_logits[remove_mask] = float('-inf')

                logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

            probs = self.softmax(logits)

            if do_sample:
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
            if use_cache:
                x_input = next_token
            else:
                x_input = torch.cat([x, next_token], dim=1)

            x = torch.cat([x, next_token], dim=1)
            
        return x
    
    def fit(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, num_epoch: int, learning_rate: float) -> None:
        self.to(self.device)
        optimizer = optim.Adam(params=self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        
        for epoch in range(num_epoch):
            
            self.train()
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch} [train]')
            for inputs, targets in train_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self(inputs)
                logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
                targets = targets.flatten()
                loss = loss_func(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(loss=loss.item())
            
            self.eval()
            valid_bar = tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{num_epoch} [valid]')
            with torch.no_grad():
                for inputs, targets in valid_bar:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    logits = self(inputs)
                    logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
                    targets = targets.flatten()
                    loss = loss_func(logits, targets)

                    valid_bar.set_postfix(loss=loss.item())
                    