import os
import numpy as np


def get_normalized_actions(env, actions: np.array, resolution: int):
    if env == 'single':
        return np.array(actions-1)
    elif env == 'portfolio':
        normalized_portfolio = (actions - ((resolution-1)/2))/((resolution-1)/2)
        end_position = sum(normalized_portfolio)
        if end_position > 1:
            normalized_portfolio = normalized_portfolio / end_position
            end_position = 1
        elif end_position < -1:
            normalized_portfolio = normalized_portfolio / abs(end_position)
            end_position = -1
        return np.array(normalized_portfolio)
    
def get_cost(actions_start, actions_end, trading_cost_bps, time_cost_bps):
    n_trades = np.sum(np.abs(actions_start - actions_end))
    time_cost = 0 if np.array_equal(actions_start, actions_end) else 1
    return n_trades * trading_cost_bps + time_cost * time_cost_bps

def get_final_model(model_path, env, model):
    if model == 'DDQN':
        model_name = f'final_model_{env}_'
        model_type = '.joblib'
    elif model == 'PPO':
        model_name = f'ppo_{env}_trading_agent_'
        model_type = '.zip'
    elif model == 'A2C':
        model_name = f'a2c_{env}_trading_agent_'
        model_type = '.zip'

    else:
        raise ValueError(f"Unknown model type {model}")
    
    model_files = [f for f in os.listdir(model_path) if f.startswith(model_name) and f.endswith(model_type)]
    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_name}")
    
    # Sort files by modification time (newest first)
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)

    # Get the latest model file
    latest_model_file = model_files[0]
    final_model_path = os.path.join(model_path, latest_model_file)
    return final_model_path


        