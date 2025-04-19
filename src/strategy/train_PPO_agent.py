
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env  
import sys
import os
proj_path = os.path.abspath(os.path.join(os.getcwd(), "../src"))
sys.path.insert(0, proj_path)


def train_PPO_agent(config, logger, env=None):
    #env = trading_env.TradingEnv(config=config, logger=logger)
    #try:
        #check_env(env)
    #except Exception as e:
        #logger.error(f"Environment check failed: {e}")
        #sys.exit(1)
    # Wrap the environment
    learning_rate = config["strategy"].get("learning_rate", 3e-4)
    total_timesteps = config["strategy"].get("total_timesteps", 100000)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)

    model.save(os.path.join(model_path, "ppo_trading_agent"))
    logger.info(f"Model saved to {model_path}")
    # Save the model 
    return model

