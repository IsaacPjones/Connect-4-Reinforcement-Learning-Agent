# pettingzoo connect 4 doc: https://pettingzoo.farama.org/environments/classic/connect_four/

from pettingzoo.classic import connect_four_v3
import torch
import torch.nn as nn
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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

def convertStateToNNInput(c4_state, agent_num):
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

def run():
  env = connect_four_v3.env(render_mode="human")
  env.reset(seed=42)

  agent1 = ConnectFourDQN()
  agent2 = ConnectFourDQN()

  for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
      action = None
    else:
      mask = observation["action_mask"]
      # this is where you would insert your policy

      if (agent == env.agents[0]):  
        q_values = agent1(convertStateToNNInput(observation['observation'], 1))
      else:
        q_values = agent2(convertStateToNNInput(observation['observation'], 2))

      # convert Tensor object to np array
      if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()

      # prevent doing any illegal actions (ex. certain columns are full)
      if mask is not None:
        q_values = np.where(mask, q_values, -np.inf)

      action = int(np.argmax(q_values))

    env.step(action)
  env.close()
run()