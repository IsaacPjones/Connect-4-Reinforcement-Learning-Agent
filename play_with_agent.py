from pettingzoo.classic import connect_four_v3
from agent import ConnectFourAgent
import torch
import numpy as np

def human_vs_agent(model_path="connect4_dqn_trained.pth"):
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    env.render()

    agent = ConnectFourAgent()
    agent.dqn.load_state_dict(torch.load(model_path, map_location="cpu"))

    done = False

    print("You are", env.agents[0], "(columns 0-6)")

    while not done:
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if reward == 1:
                    print("Winner:", agent_name)
                elif reward == -1:
                    print("Winner: other player")
                else:
                    print("Draw")
                env.step(None)
                env.render()
                done = True
                break

            action_mask = observation["action_mask"]

            if agent_name == env.agents[0]:
                valid_moves = np.where(action_mask == 1)[0]
                print("Your move, choose column from", list(valid_moves), ": ", end="")
                move = None
                while move not in valid_moves:
                    try:
                        user_input = input()
                        move = int(user_input)
                    except:
                        move = None
                env.step(int(move))
                env.render()
            else:
                q_values = agent.get_Q_values(observation)
                if isinstance(q_values, torch.Tensor):
                    q_values = q_values.detach().cpu().numpy().reshape(-1)
                masked_q = np.where(action_mask == 1, q_values, -1e9)
                action = int(np.argmax(masked_q))
                env.step(action)
                env.render()

    env.close()

if __name__ == "__main__":
    human_vs_agent()
