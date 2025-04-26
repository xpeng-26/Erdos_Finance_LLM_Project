import sys
import os

# proj_path = os.path.abspath(os.path.join(os.getcwd(), "../src"))
# sys.path.insert(0, proj_path)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)
from src.strategy.engine.trading_agent import DDQNAgent

from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
import sys
import os
from src.strategy.engine.utils_train import *

from src.strategy.engine import trading_env_portfolio
from gymnasium.spaces import Discrete, MultiDiscrete
import os
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from gymnasium.wrappers import FlattenObservation
from gymnasium import register
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import torch
from torchinfo import summary
import torch
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = config['strategy']['environment']


def backtest_DDQN_agent(config, logger, model_path):
    test_start = config["strategy"]["eval_start_date"]
    test_end = config["strategy"]["eval_end_date"]
    logger.info(f"Backtesting from {test_start} to {test_end} for DDQN agent")
    trading_days = config["strategy"]["trading_days"]
    register(
        id="trading-port-v0",
        entry_point="strategy.engine.trading_env_portfolio:TradingEnv",
        max_episode_steps=trading_days,
    )
    trading_environment = gym.make("trading-port-v0", config=config, logger=logger)

    logger.info("Environment: {portfolio}")
    seed = 42
    trading_environment.reset(seed=seed, options=None)

    if isinstance(trading_environment.action_space, spaces.Discrete):
        state_dimension = trading_environment.observation_space.shape[0]
        flattened_state_dimension = state_dimension
        action_dimension = trading_environment.action_space.n
    elif isinstance(trading_environment.action_space, spaces.MultiDiscrete):
        state_dimension = trading_environment.observation_space.shape
        flattened_state_dimension = np.prod(state_dimension)
        action_dimension = trading_environment.action_space.nvec
        # If you need the total number of possible actions
        total_actions = np.prod(action_dimension)
    else:
        raise ValueError(
            f"Unsupported action space type: {type(trading_environment.action_space)}"
        )

    agent = DDQNAgent(
        config=config,
        logger=logger,
        state_dimension=state_dimension,
        action_dimension=action_dimension,
        device=device,
    )

    logger.info(f"Loaded DDQN weights from {model_path}")

    logger.info(summary(agent.online_model, input_size=(1, flattened_state_dimension)))

    # If there is a checkpoint, load it
    # Load the latest checkpoint if it exists
    load_checkpoint(agent, model_path, device, logger, "portfolio")

    equity_curve = []
    actions = []
    done = False
    max_steps = trading_environment.spec.max_episode_steps
    state = trading_environment.unwrapped.reset_trading()
    for step in range(max_steps):
        # pick greedy action
        # state_tensor = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        # action = agent.epsilon_greedy(state_tensor, greedy=True)
        # action = agent.epsilon_greedy(state.reshape(-1, state_dimension))
        action = agent.epsilon_greedy(state.reshape(-1, flattened_state_dimension))

        actions.append(action)

        next_state, reward, done, info = trading_environment.unwrapped.trading_env_step(
            action
        )
        # next_state, reward, done, _ = trading_environment.unwrapped.trading_env_step(action)
        nav = info.get("nav")
        equity_curve.append(nav)
        logger.info(
            f"[Step {step:03d}] action={action}, reward={reward:.4f}, nav={nav:.4f}"
        )

        # execute one step
        # next_state, reward, done, _ = trading_environment.unwrapped.trading_env_step(int(action))
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        # done_val = torch.tensor(0.0 if done else 1.0, dtype=torch.float32, device=device)
        state = next_state

        if done:
            break

    # 5) At the end, grab the entire trade history from the simulator
    df = trading_environment.unwrapped.simulator.result()

    plt.figure(figsize=(10, 5))
    plt.plot(df.nav, label="DDQN NAV")
    plt.plot(df.market_nav, label="Market NAV", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Net Asset Value")
    plt.title(f"DDQN Backtest {test_start} → {test_end}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df, actions, equity_curve


def backtest_PPO_agent(config, logger, model_path):
    test_start = config["strategy"].get("test_start_date")
    test_end = config["strategy"].get("test_end_date")
    logger.info(f"Backtesting on period: {test_start} → {test_end}")
    config["strategy"]["train_start_date"] = test_start
    config["strategy"]["train_end_date"] = test_end
    env = trading_env_portfolio.TradingEnv(config=config, logger=logger)
    print(env.data_source.date_length)

    env = FlattenObservation(env)
    obs, info = env.reset()
    done = False

    equity_curve = []
    actions = []
    steps = 0
    model = PPO.load(model_path, env=env)
    while not done:
        # Get the action (deterministic for backtest)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract NAV from info (or fallback to the simulator’s internal array)
        nav = info.get("nav")
        if nav is None:
            sim = env.unwrapped.simulator
            nav = sim.navs[sim.step - 1]

        equity_curve.append(nav)
        logger.info(
            f"[Step {steps:03d}] action={action}, reward={reward:.4f}, nav={nav:.4f}"
        )
        steps += 1
        df = env.unwrapped.simulator.result()

    return df, actions, equity_curve
