# pettingzoo connect 4 doc: https://pettingzoo.farama.org/environments/classic/connect_four/

from pettingzoo.classic import connect_four_v3
import torch
import numpy as np
from agent import ConnectFourAgent
from exp_replay import ReplayMemory
import yaml
import random
import os


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
  
  def load_model_if_exists(self, agent, model_path="connect4_dqn.pth"):
    """Load model weights if the file exists"""
    if os.path.exists(model_path):
        agent.dqn.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
        return True
    else:
        print("No existing model found, starting fresh")
        return False

  def run(self, is_training=True):

    if is_training:
        env = connect_four_v3.env() #to speed up training by not rendering
    else:
        env = connect_four_v3.env(render_mode="human")

    agent = ConnectFourAgent(
      learning_rate=self.learning_rate,
      discount_factor=self.discount_factor,
      device=device
    )
    
    if is_training:
        self.load_model_if_exists(agent, model_path="connect4_dqn_trained.pth")

    # moved replay memory creation before the episode loop
    if (is_training):
        replay_memory = ReplayMemory(self.replay_memory_size)

    # use iterations to control number of episodes
    iterations = 4000
    train_steps = 0

    agent_wins = 0
    opponent_wins = 0

    for episode in range(iterations):
      env.reset() # Removed seed for more varied training

      last_observation = {env.agents[0]: None, env.agents[1]: None}
      last_action = {env.agents[0]: None, env.agents[1]: None}

      training_agent = env.agents[0]

      for agent_name in env.agent_iter():
        # note: all these values are agent-specific
        observation, reward, termination, truncation, info = env.last()
        
        if (termination or truncation) and reward != 0:
          if agent_name == training_agent and reward > 0:
            agent_wins += 1
          elif agent_name != training_agent and reward > 0:
            opponent_wins += 1

        if termination or truncation:
          action = None
        else:
          # Using the action mask to only select valid actions, because it kept going off the board
          action_mask = observation['action_mask']
          if agent_name == training_agent:
            if (random.random() < self.epsilon):
              valid_actions = np.where(action_mask == 1)[0]
              action = np.random.choice(valid_actions)
            else:
              q_values = agent.get_Q_values(observation)

              if isinstance(q_values, torch.Tensor):
                q_values = q_values.detach().cpu().numpy().flatten()

              q_values_masked = np.where(action_mask == 1, q_values, -1e9)
              action = int(np.argmax(q_values_masked))

          else:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
        env.step(action)

        if (
            is_training
            and agent_name == training_agent
            and last_observation[agent_name] is not None
            and last_action[agent_name] is not None
        ):
            prev_state_input_tensor = agent.convert_state_to_NN_input(
                last_observation[agent_name]["observation"]
            )
            cur_state_input_tensor = agent.convert_state_to_NN_input(
                observation["observation"]
            )

            action_taken = last_action[agent_name]
            shaped_reward = reward
            if action_taken == 3:
                shaped_reward += 0.05
            elif action_taken in (2, 4):
                shaped_reward += 0.02

            action_tensor = torch.tensor(action_taken, dtype=torch.int64, device=device)
            reward_tensor = torch.tensor(float(shaped_reward), dtype=torch.float32, device=device)

            replay_memory.append(
                (
                  prev_state_input_tensor.squeeze(0),
                  action_tensor,
                  reward_tensor,
                  cur_state_input_tensor.squeeze(0),
                  termination or truncation
                )
            )

            if len(replay_memory) >= self.mini_batch_size:
              loss = agent.train(replay_memory, self.mini_batch_size)
              if loss is not None:
                train_steps += 1
                if train_steps % 1000 == 0:
                  agent.update_target_network()
          

        # record the observation and action taken for the agent who just acted
        last_observation[agent_name] = observation
        last_action[agent_name] = action
    
      # making interval for printing, might change this
      log_interval = max(1, iterations // 20)

      if (episode + 1) % log_interval == 0 or (episode + 1) == iterations:
        total_games = episode + 1
        win_rate = agent_wins / total_games if total_games > 0 else 0
        
        print(f"\nEpisode {episode + 1}/{iterations}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Win Rate: {win_rate:.1%} ({agent_wins} wins / {total_games} games)")
        print(f"  Training steps: {train_steps}")
        print(f"  Replay memory size: {len(replay_memory)}")
        
        if train_steps > 0:
          print(f"  Latest loss: {loss:.4f}")

      # moved this to decay after each episode
      self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total games: {iterations}")
    print(f"Agent wins: {agent_wins} ({agent_wins/iterations:.1%})")
    print(f"Opponent wins: {opponent_wins} ({opponent_wins/iterations:.1%})")
    print(f"Final epsilon: {self.epsilon:.3f}")
    print(f"Total training steps: {train_steps}")

    torch.save(agent.dqn.state_dict(), "connect4_dqn_trained.pth")
    print("Model saved.")
    env.close()

if __name__ == "__main__":
    game = GameEnvironment("connect4")
    game.run(is_training=True)
    
    