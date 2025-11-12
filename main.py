# pettingzoo connect 4 doc: https://pettingzoo.farama.org/environments/classic/connect_four/

from pettingzoo.classic import connect_four_v3
import torch
import numpy as np
from agent import ConnectFourAgent
from exp_replay import ReplayMemory

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def run(is_training=True):
  env = connect_four_v3.env(render_mode="human")
  env.reset(seed=42)

  agent0 = ConnectFourAgent(1)
  agent1 = ConnectFourAgent(2)

  if (is_training):
    replay_memory = ReplayMemory(10000)  

  last_observation = {env.agents[0]: None, env.agents[1]: None}
  last_action = {env.agents[0]: None, env.agents[1]: None}

  for agent in env.agent_iter():
    # note: all these values are agent-specific
    observation, reward, termination, truncation, info = env.last()
    prev_observation = observation

    if termination or truncation:
      action = None
    else:
      mask = observation["action_mask"]

      if (agent == env.agents[0]):  
        q_values = agent0.get_Q_values(observation)
      else:
        q_values = agent1.get_Q_values(observation)

      # convert Tensor object to np array
      if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()

      # prevent doing any illegal actions (ex. certain columns are full)
      if mask is not None:
        q_values = np.where(mask, q_values, -np.inf)

      action = int(np.argmax(q_values))

    env.step(action)

    # add entry to experience replay
    if (is_training and not last_observation[agent] == None and len(env.agents) > 0):
      cur_agent_num = 0 if agent == env.agents[0] else 1
      opposing_agent_num = 1 - cur_agent_num

      # combines the state with that state's current player 
      prev_state_input = np.append(last_observation[agent]["observation"], cur_agent_num)
      cur_state_input = np.append(observation["observation"], opposing_agent_num)

      replay_memory.append((prev_state_input, action, reward, cur_state_input, termination or truncation))

    # record the observation and action taken for the agent who just acted
    last_observation[agent] = observation
    last_action[agent] = action


  env.close()
run()