import torch
import numpy as np
from pettingzoo.classic import connect_four_v3
from agent import ConnectFourAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent(num_games=1000):
    """
    Evaluate trained agent against random opponent.
    """
    env = connect_four_v3.env()
    agent = ConnectFourAgent(device=device)

    try:
        agent.dqn.load_state_dict(torch.load("connect4_dqn_trained.pth", map_location=device))
        agent.dqn.eval()
    except FileNotFoundError:
        print("Error: connect4_dqn_trained.pth not found. Please train the agent before evaluation.")
        return

    wins = 0
    losses = 0
    draws = 0

    print(f"Evaluating agent over {num_games} games...")
    print("Trained agent is player_0, random agent is player_1\n")

    for game_num in range(num_games):
        env.reset()
        
        game_over = False
        outcome = None  # can be win, loss or draw
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # Check if game ended and record outcome
            if (termination or truncation) and not game_over:
                game_over = True
                if agent_name == env.agents[0]:
                    if reward > 0:
                        outcome = 'win'
                    elif reward < 0:
                        outcome = 'loss'
                    else:
                        outcome = 'draw'
                else:
                    if reward > 0:
                        outcome = 'loss'
                    elif reward < 0:
                        outcome = 'win'
                    else:
                        outcome = 'draw'
            
            if termination or truncation:
                action = None
            else:
                action_mask = observation["action_mask"]
                
                if agent_name == env.agents[0]:
                    # using trained agent (no exploration)
                    with torch.no_grad():
                        q_vals = agent.get_Q_values(observation)
                        q_vals = q_vals.detach().cpu().numpy().flatten()
                        masked_q = np.where(action_mask, q_vals, -np.inf)
                        action = int(np.argmax(masked_q))
                else:
                    valid_moves = np.where(action_mask == 1)[0]
                    action = int(np.random.choice(valid_moves))

            env.step(action)
        
        if outcome == 'win':
            wins += 1
        elif outcome == 'loss':
            losses += 1
        else:
            draws += 1
        
        # Progress indicator
        if (game_num + 1) % 100 == 0:
            current_win_rate = wins / (game_num + 1)
            print(f"  Games {game_num + 1}/{num_games} - Win rate: {current_win_rate:.1%} (W:{wins} L:{losses} D:{draws})")

    env.close()
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ({num_games} games vs random)")
    print("="*50)
    print(f"Wins:   {wins:3d} ({wins/num_games:6.1%})")
    print(f"Losses: {losses:3d} ({losses/num_games:6.1%})")
    print(f"Draws:  {draws:3d} ({draws/num_games:6.1%})")
    print("="*50)
    
    return wins, losses, draws


if __name__ == "__main__":
    evaluate_agent(num_games=1000)