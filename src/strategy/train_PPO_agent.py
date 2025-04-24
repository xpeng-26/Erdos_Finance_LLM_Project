
import sys
import os
from datetime import datetime

from gymnasium import register
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env  


def train_PPO_agent(config, logger):

    # Get the model path
    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)
    result_path = os.path.join(config['info']['local_data_path'], 'evaluation')
    os.makedirs(result_path, exist_ok=True)

    # Environment parameters
    learning_rate = config["strategy"]['PPO_learning_rate']
    timesteps = config["strategy"]['PPO_time_steps']
    trading_days = config["strategy"]['trading_days']
    gamma = config["strategy"]['gamma']

    # Get the environment
    env = config['strategy']['environment']
    news = config['strategy']['news']
    # register the environment
    if env == 'single':
        register(
            id='trading-v0',
            entry_point='strategy.engine.trading_env:TradingEnv',
            max_episode_steps=trading_days
        )
        # make the environment
        trading_environment = gym.make('trading-v0', config=config, logger=logger)
    elif env == 'portfolio':
        register(
            id='trading-port-v0',
            entry_point='strategy.engine.trading_env_portfolio:TradingEnv',
            max_episode_steps=trading_days
        )
        # make the environment
        trading_environment = gym.make('trading-port-v0', config=config, logger=logger)
    
    logger.info(f'Environment: {env}, With news: {news}')
    seed = 42
    trading_environment.reset(seed = seed, options=None)

    # check the environment
    check_env(trading_environment, warn=True)
    # Define the model
    model = PPO("MlpPolicy", trading_environment, verbose=1, learning_rate=learning_rate, n_steps=trading_days, seed=seed, gamma=gamma)
    logger.info(f"Model: {model}")
    
    # Train the model
    model.learn(total_timesteps=timesteps*trading_days)

    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)

    # save the final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save(os.path.join(model_path, f"ppo_{env}_{news}_trading_agent_{timestamp}"))
    logger.info(f"Model saved to {model_path}")
  
    

def train_A2C_agent(config, logger):
    # Get the model path
    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)
    result_path = os.path.join(config['info']['local_data_path'], 'evaluation')
    os.makedirs(result_path, exist_ok=True)

    # Environment parameters
    learning_rate = config["strategy"]['A2C_learning_rate']
    timesteps = config["strategy"]['A2C_time_steps']
    trading_days = config["strategy"]['trading_days']
    gamma = config["strategy"]['gamma']

    # Get the environment
    env = config['strategy']['environment']
    news = config['strategy']['news']
    # register the environment
    if env == 'single':
        register(
            id='trading-v0',
            entry_point='strategy.engine.trading_env:TradingEnv',
            max_episode_steps=trading_days
        )
        # make the environment
        trading_environment = gym.make('trading-v0', config=config, logger=logger)
    elif env == 'portfolio':
        register(
            id='trading-port-v0',
            entry_point='strategy.engine.trading_env_portfolio:TradingEnv',
            max_episode_steps=trading_days
        )
        # make the environment
        trading_environment = gym.make('trading-port-v0', config=config, logger=logger)
    
    logger.info(f'Environment: {env}, With news: {news}')
    seed = 42
    trading_environment.reset(seed = seed, options=None)

    # check the environment
    check_env(trading_environment, warn=True)
    # Define the model
    model = A2C("MlpPolicy", trading_environment, verbose=1, learning_rate=learning_rate, n_steps=trading_days, seed=seed, gamma=gamma)
    logger.info(f"Model: {model}")

    # Train the model
    model.learn(total_timesteps=timesteps*trading_days)
    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)
    # save the final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save(os.path.join(model_path, f"a2c_{env}_{news}_trading_agent_{timestamp}"))
    logger.info(f"Model saved to {model_path}")
