
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env  
import sys
import os
proj_path = os.path.abspath(os.path.join(os.getcwd(), "../src"))
sys.path.insert(0, proj_path)


def train_PPO_agent(config, logger):

    # Get the model path
    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)
    result_path = os.path.join(config['info']['local_data_path'], 'evaluation')
    os.makedirs(result_path, exist_ok=True)

    # Environment parameters
    learning_rate = config["strategy"]['PPO_learning_rate']
    total_timesteps = config["strategy"]['PPO_total_time_step']

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)

    model.save(os.path.join(model_path, "ppo_trading_agent"))
    logger.info(f"Model saved to {model_path}")
    # Save the model 
    return model

def train_A2C_agent(config, logger, )
