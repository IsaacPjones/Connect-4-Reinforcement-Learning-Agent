# pettingzoo connect 4 doc: https://pettingzoo.farama.org/environments/classic/connect_four/

from pettingzoo.classic import connect_four_v3
import torch
import numpy as np
from agent import ConnectFourAgent
from exp_replay import ReplayMemory
import yaml
import random

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class GameEnvironment():
  def __init__(self, hyperparameter_set):
    with open('hyperparameters.yml', 'r') as f:
      all_hyperparameter_sets = yaml.safe_load(f)
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]
  
    self.replay_memory_size = hyperparameters['replay_memory_size']
    self.mini_batch_size = hyperparameters['mini_batch_size']
    self.epsilon_init = hyperparameters['epsilon_init']
    self.epsilon_decay = hyperparameters['epsilon_decay']
    self.epsilon_min = hyperparameters['epsilon_min']
    self.epsilon = self.epsilon_init

  def run(self, is_training=True):
    env = connect_four_v3.env(render_mode="human")

    agent0 = ConnectFourAgent()
    agent1 = ConnectFourAgent()

    for episode in range(5):
      env.reset(seed=42)

      if (is_training):
        replay_memory = ReplayMemory(self.replay_memory_size)  

        # these variables are currently not used, could be useful for evaluating the model?
        reward_history = []
        epsilon_history = []

      last_observation = {env.agents[0]: None, env.agents[1]: None}
      last_action = {env.agents[0]: None, env.agents[1]: None}

      for agent in env.agent_iter():
        # note: all these values are agent-specific
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
          action = None
        else:
          if (random.random() < self.epsilon):
            # takes random action (exploration)
            action = env.action_space(agent).sample()
          else:
            # takes action with max q value
            if (agent == env.agents[0]):  
              q_values = agent0.get_Q_values(observation)
            else:
              q_values = agent1.get_Q_values(observation)

            # convert Tensor object to np array
            if isinstance(q_values, torch.Tensor):
              q_values = q_values.detach().cpu().numpy()

            action = int(np.argmax(q_values))

        env.step(action)

        # add entry to experience replay
        if (is_training and not last_observation[agent] == None and len(env.agents) > 0):
          cur_agent_num = 0 if agent == env.agents[0] else 1
          opposing_agent_num = 1 - cur_agent_num

          # combines the state with that state's current player 
          prev_state_input = np.append(last_observation[agent]["observation"], cur_agent_num)
          cur_state_input = np.append(observation["observation"], opposing_agent_num)

          prev_state_input_tensor = torch.tensor(prev_state_input, dtype=torch.int64, device=device)
          action_tensor = torch.tensor(action, dtype=torch.int64, device=device) if not action == None else torch.tensor(-1, dtype=torch.int64, device=device)
          reward_tensor = torch.tensor(reward, dtype=torch.int64, device=device)
          cur_state_input_tensor = torch.tensor(cur_state_input, dtype=torch.int64, device=device)

          replay_memory.append((prev_state_input_tensor, action_tensor, reward_tensor, cur_state_input_tensor, termination or truncation))

        # record the observation and action taken for the agent who just acted
        last_observation[agent] = observation
        last_action[agent] = action
    torch.save(agent0.dqn.state_dict(), "connect4_dqn.pth")
    # decay epsilon
    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    env.close()

GameEnvironment("connect4").run()
