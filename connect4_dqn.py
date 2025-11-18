import torch
import torch.nn as nn
import numpy as np

class ConnectFourDQN(nn.Module):
  def __init__(self):
    super().__init__()

    # 6x7x2 = 84 different input parameters whose values are -1, 0 or 1
    # 2 layers hidden nodes, size 128 and 64
    # outputs 7 q-values for each of the 7 possible actions
    self.net = nn.Sequential (
      nn.Linear(84, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 7)
    )

  def forward(self, x):
    return torch.clamp(self.net(x), -10.0, 10.0)