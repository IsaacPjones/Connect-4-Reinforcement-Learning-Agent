import torch
import numpy as np
from pettingzoo.classic import connect_four_v3
from agent import ConnectFourAgent
from main import device
import yaml


def evaluate_agent(num_games=50):
    env = connect_four_v3.env()
    agent = ConnectFourAgent()

    # load trained weights
    agent.dqn.load_state_dict(torch.load("connect4_dqn_trained.pth", map_location=device))
    agent.dqn.eval()

    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        env.reset()
        current_agent = env.agents[0]
        done = False
        reward_total = 0

        while not done:
            agent_name = env.agent_selection
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                if agent_name == current_agent:
                    q_vals = agent.get_Q_values(observation)
                    q_vals = q_vals.detach().cpu().numpy()
                    action_mask = observation["action_mask"]
                    masked_q = np.where(action_mask, q_vals, -np.inf)
                    action = int(np.argmax(masked_q))
                else:
                    valid_moves = np.where(observation["action_mask"] == 1)[0]
                    action = int(np.random.choice(valid_moves))

            env.step(action)
            reward_total += reward
            done = termination or truncation

        if reward_total > 0:
            wins += 1
        elif reward_total < 0:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation results over {num_games} games:")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    evaluate_agent(num_games=50)
