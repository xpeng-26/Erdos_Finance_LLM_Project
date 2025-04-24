# Import pacakges in here
import os
import sqlite3

import pandas as pd
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


# Define the class of Data source
class DataSource:
    """
    A class to handle data source operations.
    """

    def __init__(self, config: dict, logger):
        """
        Initialize the Datasource class.
        """
        self.config = config
        self.logger = logger
        self.trading_days = self.config["strategy"]["trading_days"]
        self.ticker = self.config["strategy"]["ticker"]
        
        # Get the start and end date for training
        self.start_date = self.config["strategy"]["train_start_date"]
        self.end_date = self.config["strategy"]["train_end_date"]

        self.data = self.load_data()  # Load data from the source
        self.min_values = (
            self.data.min()
        )  # Return the min values of the data, this will be used to build the gym environment
        self.max_values = (
            self.data.max()
        )  # Return the max values of the data, this will be used to build the gym environment
        # return dataframe or any other data structure
        self.step = 0  # Initialize step counter
        self.offset = 0  # Initialize offset for data slicing

    def load_data(self):
        """
        Get data from the data source.
        """
        # local loading path
        path = os.path.join(
            self.config["info"]["local_data_path"],
            "data_raw",
            self.config["info"]["db_name"],
        )

        # Query the database
        Query = f"""SELECT * FROM daily_prices d LEFT JOIN technical_factors t 
        ON d.date = t.date WHERE t.symbol = '{self.ticker}' AND d.symbol = '{self.ticker}' 
        AND DATE(d.date) >= '{self.start_date}' AND DATE(d.date) <= '{self.end_date}' ORDER BY d.date"""

        # Run the query and return the data
        db = sqlite3.connect(path)
        df = pd.read_sql_query(Query, db)
        self.logger.info(f"Data loaded from {path}, extract ticker: {self.ticker}")
        db.close()
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
        df = df.set_index("date")
        df = df.drop(columns=["symbol"])
        df['return'] = df['close'].pct_change()
        df = df.dropna()

        return df

    def reset(self):
        """
        Provides starting index for time series and resets step
        """

        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(0, high)  # Randomly select a starting point
        self.step = 0  # Reset step counter

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        market_return = self.data.iloc[self.offset + self.step]["return"]
        self.step += 1
        done = self.step > self.trading_days
        return obs, market_return, done


# Define the class of Trading simulator
class TradingSimulator:
    def __init__(self, config: dict, logger):
        # invariant for object lifetime
        self.config = config
        self.logger = logger
        self.steps = self.config["strategy"]["trading_days"]
        self.trading_cost_bps = self.config["strategy"]["trading_cost_bps"]
        self.time_cost_bps = self.config["strategy"]["time_cost_bps"]
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.zeros(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        """
        Reset the trading simulator to its initial state.
        """
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """Calculates NAVs (Net Asset Value), trading costs and reward
        based on an action and latest market return
        and returns the reward and a summary of the day's activity."""
        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action
        self.trades[self.step] = 0
        # Update position
        end_position = action - 1  # short, neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # Rough value
        trading_cost = self.trading_cost_bps * abs(n_trades)
        time_cost = 0 if n_trades != 0 else self.time_cost_bps
        self.costs[self.step] = trading_cost + time_cost
        reward = start_position * market_return - self.costs[max(0, self.step - 1)]
        self.strategy_returns[self.step] = reward

        # Update NAVs
        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (
                1 + self.market_returns[self.step]
            )

        info = {
            "reward": reward,
            "nav": self.navs[self.step],
            "costs": self.costs[self.step],
        }

        self.step += 1
        return reward, info

    def result(self):
        """
        Get the result of the trading simulator.
        """
        return pd.DataFrame(
            {
                "action": self.actions,  # current action
                "nav": self.navs,  # starting Net Asset Value (NAV)
                "market_nav": self.market_navs,
                "market_return": self.market_returns,
                "strategy_return": self.strategy_returns,
                "position": self.positions,  # eod position
                "cost": self.costs,  # eod costs
                "trade": self.trades,
            }
        )  # eod trade)


# Define the class of Trading environment
class TradingEnv(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG
    # The agent can take a long position in the stock.

    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins. This indicates a successful 
    trading episode for the agent.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.data_source = DataSource(config=self.config, logger=self.logger)
        
        self.simulator = TradingSimulator(config=self.config, logger=self.logger)
        self.action_space = spaces.Discrete(3)
        
        # Define the observation space
        self.observation_space = spaces.Box(
            low=self.data_source.min_values.values,
            high=self.data_source.max_values.values,
            shape=(len(self.data_source.min_values),),  # Proper shape based on feature count
            dtype=np.float32,
        )

    def seeding(self, seed=None):
        """
        Set the random seed for the environment.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def trading_env_step(self, action):
        """
        Execute one time step within the environment.
        """
        assert self.action_space.contains(action), "{} {} invalid".format(
            action, type(action)
        )
        observation, market_return, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(
            action=action, market_return=market_return
        )
        return observation, reward, done, info

    def reset_trading(self):
        """
        Resets the trading environment by resetting the DataSource and TradingSimulator.

        This method initializes the trading environment to its starting state by resetting
        both the data source and the trading simulator. After resetting, it retrieves and 
        returns the first observation from the data source.

        Returns:
            Any: The first observation from the data source after resetting.
        """
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]
    
    ############
    # These functions are for the stable_baselines3
    def reset(self, *, seed: int = None, options: dict = None):
   
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        else:
            self.np_random, _ = seeding.np_random()

    
        self.data_source.reset()
        self.simulator.reset()

    #
        obs, _, _ = self.data_source.take_step()
        obs = np.array(obs, dtype=np.float32)

   
        return obs, {} 
    
    def step(self, action):
        # Execute one time step using your existing logic.
        obs, reward, done, info = self.trading_env_step(action)
        obs = np.array(obs, dtype=np.float32)
    
        # Map your 'done' flag to 'terminated', and assume no truncation.
        terminated = done      # 'terminated' reflects that an episode ended naturally.
        truncated = False      # 'truncated' could be used if you implement time limits, etc.
    
        # Return five values as required by Gymnasium: (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
    

    def render(self, mode="human"):
        """
        Render the environment.
        """
        pass
