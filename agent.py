import torch
import numpy as np
from connect4_dqn import ConnectFourDQN


class ConnectFourAgent():
  def __init__(self, agent_num):
    self.dqn = ConnectFourDQN()
    self.agent_num = agent_num

  def convert_state_to_NN_input(self, c4_state, agent_num):
    """
    Convert a (6, 7, 2) array with values 0 and 1 into a (6, 7) array with values -1, 0, and 1.
    Depending on the agent calling this function, converts the opposing agent's 1 values to -1 in the reformed array

    Params:
    c4_state - (6, 7, 2) array
    agent_num - 1 or 2

    Returns:
    Tensor array of shape (6, 7)
    """
        
    main_agent_i = agent_num - 1
    other_agent_i = 1 - main_agent_i

    result = np.zeros((6, 7), dtype=int)
    result[c4_state[:, :, main_agent_i] == 1] = 1
    result[c4_state[:, :, other_agent_i] == 1] = -1

    return torch.from_numpy(result.reshape(1, -1)).float()
  
  def get_Q_values(self, observation):
    return self.dqn(self.convert_state_to_NN_input(observation['observation'], self.agent_num))
