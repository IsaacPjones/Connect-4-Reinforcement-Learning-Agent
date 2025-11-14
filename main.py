# pettingzoo connect 4 doc: https://pettingzoo.farama.org/environments/classic/connect_four/

from pettingzoo.classic import connect_four_v3
import torch
import numpy as np
from agent import ConnectFourAgent
from exp_replay import ReplayMemory
import yaml
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    self.learning_rate = hyperparameters['learning_rate']
    self.discount_factor = hyperparameters['discount_factor']

  def run(self, is_training=True):

    if is_training:
        env = connect_four_v3.env() #to speed up training by not rendering
    else:
        env = connect_four_v3.env(render_mode="human")

    agent0 = ConnectFourAgent(
        learning_rate=self.learning_rate,
        discount_factor=self.discount_factor,
        device=device
    )

    agent1 = agent0  # both players use the same agent


    # moved replay memory creation before the episode loop
    if (is_training):
        replay_memory = ReplayMemory(self.replay_memory_size)

    # use iterations to control number of episodes
    iterations = 10000
    train_steps = 0
    for episode in range(iterations):
      env.reset(seed=42)

      last_observation = {env.agents[0]: None, env.agents[1]: None}
      last_action = {env.agents[0]: None, env.agents[1]: None}

      for agent in env.agent_iter():
        # note: all these values are agent-specific
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
          action = None
        else:
          # Using the action mask to only select valid actions, because it kept going off the board
          action_mask = observation['action_mask']

          if (random.random() < self.epsilon):
            # takes random action from valid actions (exploration)
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
          else:
            # takes action with max q value
            if (agent == env.agents[0]):  
              q_values = agent0.get_Q_values(observation)
            else:
              q_values = agent1.get_Q_values(observation)

            # convert Tensor object to np array
            if isinstance(q_values, torch.Tensor):
              q_values = q_values.detach().cpu().numpy()

            q_values_masked = np.where(action_mask, q_values, -np.inf)
            action = int(np.argmax(q_values_masked))

        env.step(action)

        # add entry to experience replay
        if (is_training and not last_observation[agent] == None and len(env.agents) > 0):

          prev_state_input_tensor = agent0.convert_state_to_NN_input(last_observation[agent]["observation"])
          cur_state_input_tensor = agent0.convert_state_to_NN_input(observation["observation"])

          if action is not None:
            action_tensor = torch.tensor(action, dtype=torch.int64, device=device)
            reward_tensor = torch.tensor(np.clip(float(reward), -1, 1), dtype=torch.float32, device=device)
            replay_memory.append(
                (
                  prev_state_input_tensor.squeeze(0),
                  action_tensor,
                  reward_tensor,
                  cur_state_input_tensor.squeeze(0),
                  termination or truncation
                )
            )
            
          if len(replay_memory) > 5:  
            loss = agent0.train(replay_memory, self.mini_batch_size)
            train_steps += 1

            if train_steps % 1000 == 0:
                agent0.update_target_network()


        # record the observation and action taken for the agent who just acted
        last_observation[agent] = observation
        last_action[agent] = action
    
      # making interval for printing, might change this
      log_interval = max(1, iterations // 10)

      if (episode + 1) % log_interval == 0 or (episode + 1) == iterations:  
        print(f"Episode {episode + 1}/{iterations} - Epsilon: {self.epsilon:.3f}")

        if is_training and len(replay_memory) > 5 and loss is not None:
          print(f"Latest loss: {loss}")

      # moved this to decay after each episode
      self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    torch.save(agent0.dqn.state_dict(), "connect4_dqn_trained.pth")
    print("Model saved.")
    env.close()

if __name__ == "__main__":
    game = GameEnvironment("connect4")
    game.run(is_training=True)
    
    