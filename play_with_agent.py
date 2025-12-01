from pettingzoo.classic import connect_four_v3
from agent import ConnectFourAgent
import torch
import numpy as np
import os
import pygame

def human_vs_agent(model_path=None):
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "connect4_dqn_trained.pth")
    
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    env.render()

    agent = ConnectFourAgent()
    agent.dqn.load_state_dict(torch.load(model_path, map_location="cpu"))

    done = False

    print("You are", env.agents[0], "(columns 0-6)")
    print("Click on the column where you want to drop your piece!")

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
                print("Your turn! Valid columns:", list(valid_moves))
                
                move = None
                while move is None or move not in valid_moves:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            
                            window = pygame.display.get_surface()
                            window_width = window.get_width()
                            
                            column_width = window_width / 7
                            clicked_column = int(mouse_x / column_width)
                            
                            if 0 <= clicked_column <= 6 and clicked_column in valid_moves:
                                move = clicked_column
                                print(f"You played column {move}")
                            else:
                                print(f"Invalid column! Choose from {list(valid_moves)}")
                    
                    pygame.time.wait(10)
                
                env.step(int(move))
                env.render()
                
            else:
                q_values = agent.get_Q_values(observation)
                if isinstance(q_values, torch.Tensor):
                    q_values = q_values.detach().cpu().numpy().reshape(-1)
                masked_q = np.where(action_mask == 1, q_values, -1e9)
                action = int(np.argmax(masked_q))
                print(f"AI played column {action}")
                env.step(action)
                env.render()
                
                pygame.time.wait(500)

    env.close()

if __name__ == "__main__":
    human_vs_agent()
