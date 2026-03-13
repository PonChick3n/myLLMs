import torch
import torch.utils.data as data


class GetData(data.Dataset):
    
    def __init__(self, data: torch.Tensor, seq_len: int, device: str='cpu') -> None:
        super().__init__()
        
        self.data = data
        self.seq_len = seq_len
        self.device = device
        
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx:idx + self.seq_len])
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.seq_len])
        return x, y
    