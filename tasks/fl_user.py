from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader



@dataclass
class FLUser:
    user_id: int = 0
    compromised: bool = False
    train_loader: DataLoader = None
    backdoor_label: int = None
    number_of_samples: int = 0
