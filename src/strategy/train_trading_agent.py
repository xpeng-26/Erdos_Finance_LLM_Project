# Description: This file contains the function to train the trading agent.
from .engine.trading_agent import DDQNAgent
from .engine.utils_train import *

import os
from time import time
from datetime import datetime


from gymnasium import register
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import torch
from torchinfo import summary


####################################################################################################
# Define the training function
def train_trading_agent(config, logger):
    """
    Train the trading agent.
    """

    # Get the model path
    model_path = os.path.join(config["info"]["local_data_path"], "models")
    os.makedirs(model_path, exist_ok=True)
    result_path = os.path.join(config["info"]["local_data_path"], "evaluation")
    os.makedirs(result_path, exist_ok=True)

    # Get the configuration parameters
    trading_days = config["strategy"]["trading_days"]
    max_episodes = config["strategy"]["max_episodes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the environment
    env = config["strategy"]["environment"]
    news = config["strategy"]["news"]
    # register the environment
    if env == "single":
        register(
            id="trading-v0",
            entry_point="strategy.engine.trading_env:TradingEnv",
            max_episode_steps=trading_days,
        )
        # make the environment
        trading_environment = gym.make("trading-v0", config=config, logger=logger)
    elif env == "portfolio":
        register(
            id="trading-port-v0",
            entry_point="strategy.engine.trading_env_portfolio:TradingEnv",
            max_episode_steps=trading_days,
        )
        # make the environment
        trading_environment = gym.make("trading-port-v0", config=config, logger=logger)

    logger.info(f"Environment: {env} With news: {news}")
    seed = 42
    trading_environment.reset(seed=seed, options=None)

    # Get environment parameters
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
    max_episode_steps = trading_environment.spec.max_episode_steps

    # Create the trading agent
    trading_agent = DDQNAgent(
        config=config,
        logger=logger,
        state_dimension=state_dimension,
        action_dimension=action_dimension,
        device=device,
    )
    logger.info(
        summary(trading_agent.online_model, input_size=(1, flattened_state_dimension))
    )

    # If there is a checkpoint, load it
    # Load the latest checkpoint if it exists
    start_episode = load_checkpoint(
        trading_agent, model_path, device, logger, env, news
    )

    # Initialize the variables
    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    # Define the visualization function
    def track_results(
        episode,
        nav_ma_100,
        nav_ma_10,
        market_nav_100,
        market_nav_10,
        win_ratio,
        total,
        epsilon,
    ):
        time_ma = np.mean([episode_time[-100:]])
        T = np.sum(episode_time)

        template = "{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | "
        template += "Market: {:>6.1%} ({:>6.1%}) | "
        template += "Wins: {:>5.1%} | eps: {:>6.3f}"
        logger.info(
            template.format(
                episode,
                format_time(total),
                nav_ma_100 - 1,
                nav_ma_10 - 1,
                market_nav_100 - 1,
                market_nav_10 - 1,
                win_ratio,
                epsilon,
            )
        )

    # Train the agent
    start = time()
    result = []
    for episode in range(start_episode, max_episodes + 1):
        this_state = trading_environment.unwrapped.reset_trading()
        for episode_step in range(max_episode_steps):
            action = trading_agent.epsilon_greedy(
                this_state.reshape(-1, flattened_state_dimension)
            )

            next_state, reward, done, _ = (
                trading_environment.unwrapped.trading_env_step(action)
            )

            # Convert next_state, reward, done to tensors on the same device
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            done_val = torch.tensor(
                0.0 if done else 1.0, dtype=torch.float32, device=device
            )

            # Update the agent
            trading_agent.memorize_transition(
                this_state, action, reward, next_state, done_val
            )
            if trading_agent.train:
                trading_agent.experience_replay()
            if done:
                break
            this_state = next_state

        # get DataFrame with seqence of actions, returns and nav values
        result = trading_environment.unwrapped.simulator.result()

        # get results of last step
        final = result.iloc[-1]

        # apply return (net of cost) of last action to last starting nav
        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        # market nav
        market_nav = final.market_nav
        market_navs.append(market_nav)

        # track difference between agent an market NAV results
        diff = nav - market_nav
        diffs.append(diff)

        # display results
        if episode % 10 == 0:
            track_results(
                episode,
                np.mean(navs[-100:]),
                np.mean(navs[-10:]),
                np.mean(market_navs[-100:]),
                np.mean(market_navs[-10:]),
                np.mean([diff > 0 for diff in diffs[-100:]]),
                time() - start,
                trading_agent.epsilon,
            )

        # if agent has been winning for a while, stop training
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            # print the tail of the result
            logger.info(
                f"The agent has been winning for a while, stop training, the tail is {result.tail()}"
            )
            break

        # Save the checkpoint every 10 episodes
        if episode % 10 == 0:
            save_checkpoint(
                trading_agent,
                episode,
                config,
                model_path,
                navs,
                market_navs,
                diffs,
                logger,
                env,
                news,
            )

    logger.info(f"Finished training after {format_time(time() - start)}")
    # save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(
        trading_agent.online_model.state_dict(),
        os.path.join(model_path, f"final_model_{env}_{news}_{timestamp}.pth"),
    )
    # save the final results
    results = pd.DataFrame(
        {
            "Episode": list(range(1, len(navs) + 1)),
            "Agent": navs,
            "Market": market_navs,
            "Difference": diffs,
        }
    )
    # Add rolling strategy win percentage
    results["Strategy Wins (%)"] = (results.Difference > 0).rolling(100).sum()

    results.to_csv(
        os.path.join(result_path, f"final_results_{env}_{news}_{timestamp}.csv")
    )

    # Close the environment
    trading_environment.close()


###################################################################################################
# dump the latest final model
def dump_final_DDQN(config, logger):

    # Get the model path
    model_path = os.path.join(config["info"]["local_data_path"], "models")
    os.makedirs(model_path, exist_ok=True)

    # Get the configuration parameters
    trading_days = config["strategy"]["trading_days"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the environment
    env = config["strategy"]["environment"]
    news = config["strategy"]["news"]
    # register the environment
    if env == "single":
        register(
            id="trading-v0",
            entry_point="strategy.engine.trading_env:TradingEnv",
            max_episode_steps=trading_days,
        )
        # make the environment
        trading_environment = gym.make("trading-v0", config=config, logger=logger)
    elif env == "portfolio":
        register(
            id="trading-port-v0",
            entry_point="strategy.engine.trading_env_portfolio:TradingEnv",
            max_episode_steps=trading_days,
        )
        # make the environment
        trading_environment = gym.make("trading-port-v0", config=config, logger=logger)

    logger.info(f"Environment: {env}, With news: {news}")

    # Get environment parameters
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

    # Create the trading agent
    trading_agent = DDQNAgent(
        config=config,
        logger=logger,
        state_dimension=state_dimension,
        action_dimension=action_dimension,
        device=device,
    )
    logger.info(
        summary(trading_agent.online_model, input_size=(1, flattened_state_dimension))
    )
    # Load the latest checkpoint if it exists
    load_final_model(trading_agent, model_path, device, logger, env, news)
    # save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        trading_agent.online_model,
        os.path.join(model_path, f"final_model_{env}_{news}_{timestamp}.joblib"),
    )
    logger.info(
        f'Final model saved to {os.path.join(model_path, f"final_model_{env}_{news}_{timestamp}.joblib")}'
    )
    # Close the environment
    trading_environment.close()
    logger.info(f"Finished dumping final model")
