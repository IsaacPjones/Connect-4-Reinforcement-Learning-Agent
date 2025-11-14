import torch
import torch.nn as nn
import numpy as np
from connect4_dqn import ConnectFourDQN

class ConnectFourAgent():
    def __init__(self, learning_rate=0.0001, discount_factor=0.99, device="cpu"):
        self.device = device
        self.learning_rate = learning_rate
        self.dqn = ConnectFourDQN().to(device)
        self.discount_factor = discount_factor

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)

        # target network
        self.target_dqn = ConnectFourDQN().to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()



    # update target network
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    # reshape board to 84 input features
    def convert_state_to_NN_input(self, c4_state):
        return torch.from_numpy(c4_state.reshape(1, -1)).float().to(self.device)

    # forward pass
    def get_Q_values(self, observation):
        x = self.convert_state_to_NN_input(observation['observation'])
        return self.dqn(x)

    # training step
    def train(self, replay_memory, batch_size=32):
        if len(replay_memory) < batch_size:
            return None

        batch = replay_memory.sample(batch_size)

        states = torch.stack([s for (s, _, _, _, _) in batch]).float().to(self.device)
        actions = torch.stack([a for (_, a, _, _, _) in batch]).long().to(self.device)
        rewards = torch.stack([r for (_, _, r, _, _) in batch]).float().to(self.device)
        next_states = torch.stack([ns for (_, _, _, ns, _) in batch]).float().to(self.device)
        dones = torch.tensor([d for (_, _, _, _, d) in batch], dtype=torch.bool).to(self.device)

        # main Q network
        current_q_values = self.dqn(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target network values
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + self.discount_factor * max_next_q * (~dones)
            target_q = torch.clamp(target_q, -10, 10)


        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0) # gradient clipping to keep network stable
        self.optimizer.step()

        return loss.item()
