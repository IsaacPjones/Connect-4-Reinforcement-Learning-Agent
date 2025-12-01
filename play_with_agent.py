from pettingzoo.classic import connect_four_v3
from agent import ConnectFourAgent
import torch
import numpy as np
import threading
import queue
import time

class NonBlockingInput:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.input_thread = None
        self.stop_thread = False
        
    def start_input_thread(self, prompt, valid_moves):
        def get_input():
            print(prompt, list(valid_moves), ": ", end="", flush=True)
            while not self.stop_thread:
                try:
                    user_input = input()
                    self.input_queue.put(user_input)
                    break
                except:
                    pass
        
        self.input_thread = threading.Thread(target=get_input, daemon=True)
        self.input_thread.start()
    
    def get_input(self, timeout=0.1):
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        self.stop_thread = True

def human_vs_agent(model_path="connect4_dqn_trained.pth"):
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    env.render()

    agent = ConnectFourAgent()
    agent.dqn.load_state_dict(torch.load(model_path, map_location="cpu"))

    done = False
    input_handler = NonBlockingInput()
    waiting_for_input = False
    valid_moves = None

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
                
                if not waiting_for_input:
                    input_handler.start_input_thread("Your move, choose column from", valid_moves)
                    waiting_for_input = True
                
                move = None
                while move not in valid_moves:
                    user_input = input_handler.get_input(timeout=0.05)
                    
                    if user_input is not None:
                        try:
                            move = int(user_input)
                            if move not in valid_moves:
                                print(f"Invalid move {move}. Choose from", list(valid_moves), ": ", end="", flush=True)
                                input_handler = NonBlockingInput()
                                input_handler.start_input_thread("", valid_moves)
                        except ValueError:
                            print(f"Invalid input. Choose from", list(valid_moves), ": ", end="", flush=True)
                            input_handler = NonBlockingInput()
                            input_handler.start_input_thread("", valid_moves)
                    
                    time.sleep(0.05)
                
                waiting_for_input = False
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

    input_handler.stop()
    env.close()

if __name__ == "__main__":
    human_vs_agent()
