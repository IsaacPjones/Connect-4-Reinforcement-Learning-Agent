import torch
import numpy as np
from connect4_dqn import ConnectFourDQN
import yaml

class ConnectFourAgent():
  def __init__(self):
    self.dqn = ConnectFourDQN()

  def convert_state_to_NN_input(self, c4_state):
    """
    Convert a (6, 7, 2) array with values 0 and 1 into a (6, 7) array with values -1, 0, and 1.
    1 = Current agent's game pieces
    0 = No game pieces
    -1 = Opposing agent's game pieces

    Params:
    c4_state - (6, 7, 2) array

    Returns:
    Tensor array of shape (6, 7)
    """

    result = np.zeros((6, 7), dtype=int)
    result[c4_state[:, :, 0] == 1] = 1
    result[c4_state[:, :, 1] == 1] = -1

    return torch.from_numpy(result.reshape(1, -1)).float()
  
  def get_Q_values(self, observation):
    return self.dqn(self.convert_state_to_NN_input(observation['observation'] ))
