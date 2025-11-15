import torch
import torch.nn as nn

class TinyChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(832, 256), # input 852, out 256
            nn.ReLU(), # ReLU = Rectified Linear Unit activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # final evaluation scalar
        )

    def forward(self, x):
        return self.net(x)
