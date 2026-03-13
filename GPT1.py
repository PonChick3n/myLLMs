import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from embeddings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder


class GPT(nn.Module):
       
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int, head_size: int, num_layers: int,
                 dropout: float=0.1, device: str='cpu') -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.decoders = nn.Sequential(*[Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(emb_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x.shape[1])
        emb = token_emb + pos_emb
        emb = self.dropout(emb)
        out = self.decoders(emb)
        out = self.linear(out)
        
        return out
    
    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool, top_k: int=None, top_p: float=None,
                 temperature: float=1.0) -> torch.Tensor:
        input_seq_len = x.shape[1]
        full_sequence = x.clone()
    
        if input_seq_len > self.max_seq_len:
            current_input = x[:, -self.max_seq_len:]
        else:
            current_input = x
    
        for _ in range(max_new_tokens):
            logits = self.forward(current_input)
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
                
            full_sequence = torch.cat([full_sequence, next_token], dim=1)
            current_input = torch.cat([current_input, next_token], dim=1)
        
            if current_input.shape[1] > self.max_seq_len:
                current_input = current_input[:, -self.max_seq_len:]
    
        return full_sequence
    
    def fit(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, num_epoch: int, learning_rate: float) -> None:
        self.to(self.device)
        optimizer = optim.Adam(params=self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        
        train_losses = []
        valid_losses = []
        for _ in range(num_epoch):
            
            self.train()
            for inputs, targets in train_loader:
                logits = self(inputs)
                logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
                targets = targets.flatten()
                loss = loss_func(logits, targets)
                train_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    logits = self(inputs)
                    logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
                    targets = targets.flatten()
                    loss = loss_func(logits, targets)
                    valid_losses.append(loss.item())
                    