import os
from datetime import datetime

import numpy as np
import pandas as pd

import joblib
from stable_baselines3 import PPO, A2C
import torch

from .engine.evaluation_env import make_env
from .engine.utils import *

def evaluation_main(config, logger):

    # Get the model path
    model_path = os.path.join(config['info']['local_data_path'], 'models')
    os.makedirs(model_path, exist_ok=True)
    result_path = os.path.join(config['info']['local_data_path'], 'evaluation')
    os.makedirs(result_path, exist_ok=True)
    # Get the env
    env = config['strategy']['environment']
    news = config['strategy']['news']
    if env == 'single':
        logger.info("Single environment")
        ticker_num = 1
        resolution = 0
    elif env == 'portfolio':
        logger.info("Portfolio environment")
        ticker_num = len(config["strategy"]["tickers"])
        resolution = config["strategy"]["resolution"]
    else:
        raise ValueError(f"Unknown env type {env}")
    
    trading_cost_bps = config["strategy"]["trading_cost_bps"]
    time_cost_bps = config["strategy"]["time_cost_bps"]
    # Load the test environment
    datasorce = make_env(config, logger)
    trading_days = datasorce.trading_days
    logger.info(f"Trading days: {trading_days}")

    # Load the saved DDQN model
    Found_DDQN = False
    try:
        DDQN_path = get_final_model(model_path, env, news,'DDQN')
        model_DDQN = joblib.load(DDQN_path)
        Found_DDQN = True
        DDQN_actions = np.zeros((trading_days, ticker_num))
        DDQN_navs = np.ones(trading_days)
        DDQN_returns = np.zeros(trading_days)
        logger.info(f"DDQN model loaded from {model_path}")
    except:
        logger.info(f"DDQN model not found in {model_path}")
    
    # Load the saved PPO model
    Found_PPO = False
    try:
        PPO_path = get_final_model(model_path, env, news, 'PPO')
        model_PPO = PPO.load(PPO_path)
        Found_PPO = True
        PPO_actions = np.zeros((trading_days, ticker_num))
        PPO_navs = np.ones(trading_days)
        PPO_returns = np.zeros(trading_days)
        logger.info(f"PPO model loaded from {model_path}")
    except:
        logger.info(f"PPO model not found in {model_path}")

    # Load the saved A2C model
    Found_A2C = False
    try:
        A2C_path = get_final_model(model_path, env, news, 'A2C')
        model_A2C = A2C.load(A2C_path)
        Found_A2C = True
        A2C_actions = np.zeros((trading_days, ticker_num))
        A2C_navs = np.ones(trading_days)
        A2C_returns = np.zeros(trading_days)
        logger.info(f"A2C model loaded from {model_path}")
    except:
        logger.info(f"A2C model not found in {model_path}")


    market_navs = np.ones(trading_days)
    market_returns = np.zeros(trading_days)
    # Backtest
    datasorce.reset()
    logger.info("Backtest started")
    for day in range(trading_days):
        obs, market_return = datasorce.take_step()
        if day == 0:
            continue
        if env == 'single':
            market_returns[day] = market_return
            market_navs[day] = market_navs[day - 1] * (1 + market_return)
        elif env == 'portfolio':
            market_returns[day] = np.mean(market_return)
            market_navs[day] = market_navs[day - 1] * (1 + np.mean(market_return))
        else:
            raise ValueError(f"Unknown env type {env}")
        
        if Found_DDQN:
            if env == 'single':
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                q_values = model_DDQN(obs_tensor)
                action_DDQN_original = q_values.argmax().item()
            elif env == 'portfolio':
                obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
                q_values = model_DDQN(obs_tensor.unsqueeze(0))
                action_DDQN_original = [q.argmax().item() for q in q_values]
            action_DDQN = get_normalized_actions(env, np.array(action_DDQN_original), resolution)
            DDQN_actions[day] = action_DDQN
            return_DDQN = np.dot(DDQN_actions[day-1], market_return) - get_cost(DDQN_actions[day-1], action_DDQN, trading_cost_bps, time_cost_bps)
            DDQN_navs[day] = DDQN_navs[day - 1] * (1 + return_DDQN)
            DDQN_returns[day] = return_DDQN
        if Found_PPO:
            action_PPO = get_normalized_actions(env, np.array(model_PPO.predict(obs, deterministic=True)[0]), resolution)
            PPO_actions[day] = action_PPO
            #PPO_actions[day] = model_PPO.predict(obs, deterministic=True)[0]
            return_PPO = np.dot(PPO_actions[day-1], market_return) - get_cost(PPO_actions[day-1], action_PPO, trading_cost_bps, time_cost_bps)
            PPO_navs[day] = PPO_navs[day - 1] * (1 + return_PPO)
            PPO_returns[day] = return_PPO
        if Found_A2C:
            action_A2C = get_normalized_actions(env, np.array(model_A2C.predict(obs, deterministic=True)[0]), resolution)
            A2C_actions[day] = action_A2C
            #A2C_actions[day] = model_A2C.predict(obs, deterministic=True)[0]
            return_A2C = np.dot(A2C_actions[day-1], market_return) - get_cost(A2C_actions[day-1], action_A2C, trading_cost_bps, time_cost_bps)
            A2C_navs[day] = A2C_navs[day - 1] * (1 + return_A2C)
            A2C_returns[day] = return_A2C


    # Log the results
    logger.info("Backtest finished")
    # Save the results
    result = pd.DataFrame()
    result['market_navs'] = market_navs
    result['market_returns'] = market_returns
    if env == 'single':
        if Found_DDQN:
            result['DDQN_navs'] = DDQN_navs
            result['DDQN_returns'] = DDQN_returns
            result['DDQN_actions'] = DDQN_actions
        if Found_PPO:
            result['PPO_navs'] = PPO_navs
            result['PPO_returns'] = PPO_returns
            result['PPO_actions'] = PPO_actions
        if Found_A2C:
            result['A2C_navs'] = A2C_navs
            result['A2C_returns'] = A2C_returns
            result['A2C_actions'] = A2C_actions
    elif env == 'portfolio':
        if Found_DDQN:
            result['DDQN_navs'] = DDQN_navs
            result['DDQN_returns'] = DDQN_returns
            for i in range(ticker_num):
                result[f'DDQN_{i}'] = DDQN_actions[:, i]
        if Found_PPO:
            result['PPO_navs'] = PPO_navs
            result['PPO_returns'] = PPO_returns
            for i in range(ticker_num):
                result[f'PPO_{i}'] = PPO_actions[:, i]
        if Found_A2C:
            result['A2C_navs'] = A2C_navs
            result['A2C_returns'] = A2C_returns
            for i in range(ticker_num):
                result[f'A2C_{i}'] = A2C_actions[:, i]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Save the result
    result.to_csv(os.path.join(result_path, f'final_results_{env}_{news}_{timestamp}.csv'))
    logger.info(f"Results saved to {result_path}")

