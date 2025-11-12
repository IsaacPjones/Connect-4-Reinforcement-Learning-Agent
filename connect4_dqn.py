import torch
import torch.nn as nn
import numpy as np

class ConnectFourDQN(nn.Module):
  def __init__(self):
    super().__init__()

    # 6x7 = 42 different input parameters whose values are -1, 0 or 1
    # 2 layers hidden nodes, size 128 and 64
    # outputs 7 q-values for each of the 7 possible actions
    self.net = nn.Sequential (
      nn.Linear(42, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 7)
    )

  def forward(self, x):
    return self.net(x)