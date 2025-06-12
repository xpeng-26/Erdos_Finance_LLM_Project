import os
import sqlite3
import pandas as pd
import numpy as np


class DataSorce_single:
    def __init__(self, config: dict, logger):
        """
        Initialize the Datasource class.
        """
        self.config = config
        self.logger = logger
        self.ticker = self.config["strategy"]["ticker"]

        # Get the start and end date for training
        self.start_date = self.config["strategy"]["eval_start_date"]
        self.end_date = self.config["strategy"]["eval_end_date"]

        self.data, self.date = self.load_data()  # Load data from the source
        self.trading_days = self.data.shape[0]  # Get the number of trading days

        self.step = 0  # Initialize the step counter

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
        news = self.config["strategy"]["news"]
        # Query the database
        if news:
            Query = f"""
            SELECT 
                d.*, 
                t.*, 
                n.sentiment_short, n.sentiment_mid, n.sentiment_long, n.adjustment_mid, n.adjustment_long 
            FROM 
                daily_prices d 
            LEFT JOIN 
                technical_factors t 
            ON 
                d.date = t.date AND d.symbol = t.symbol 
            LEFT JOIN 
                news_factors n 
            ON 
                d.date = n.date AND d.symbol = n.symbol 
            WHERE 
                d.symbol = '{self.ticker}' 
                AND DATE(d.date) >= '{self.start_date}' 
                AND DATE(d.date) <= '{self.end_date}' 
            ORDER BY 
                d.date
            """
        else:
            Query = f"""SELECT * FROM daily_prices d LEFT JOIN technical_factors t 
            ON d.date = t.date WHERE t.symbol = '{self.ticker}' AND d.symbol = '{self.ticker}' 
            AND DATE(d.date) >= '{self.start_date}' AND DATE(d.date) <= '{self.end_date}' ORDER BY d.date"""
        # Run the query and return the data
        db = sqlite3.connect(path)
        df = pd.read_sql_query(Query, db)
        self.logger.info(f"Data loaded from {path}, extract ticker: {self.ticker}")
        db.close()
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
        date = df["date"][1:]
        df = df.set_index("date")
        df = df.drop(columns=["symbol"])
        df["return"] = df["close"].pct_change()
        df = df.dropna()

        return df, date

    def take_step(self):
        obs = self.data.iloc[self.step].values
        market_return = np.array(self.data.iloc[self.step]["return"])
        self.step += 1
        return obs, market_return

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.step = 0


class DataSorce_portfolio:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.tickers = self.config["strategy"]["tickers"]
        self.tickers_num = len(self.tickers)

        self.start_date = self.config["strategy"]["eval_start_date"]
        self.end_date = self.config["strategy"]["eval_end_date"]

        self.data, self.date = self.load_data()  # Load data from the source
        self.trading_days = (
            self.data["date_seq"].max() - self.data["date_seq"].min() + 1
        )

        self.step = 0  # Initialize the step counter

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
        news = self.config["strategy"]["news"]
        # Query the database
        tickers = "', '".join(self.tickers)
        if news:
            Query = f"""
            SELECT 
                DISTINCT d.*, 
                t.*, 
                n.sentiment_short, n.sentiment_mid, n.sentiment_long, n.adjustment_mid, n.adjustment_long 
            FROM 
                daily_prices d 
            LEFT JOIN 
                technical_factors t 
            ON 
                d.date = t.date AND d.symbol = t.symbol 
            LEFT JOIN 
                news_factors n 
            ON 
                d.date = n.date AND d.symbol = n.symbol 
            WHERE
                t.symbol IN ('{tickers}') 
                AND DATE(d.date) >= '{self.start_date}' 
                AND DATE(d.date) <= '{self.end_date}' 
            ORDER BY 
                d.date
            """
        else:
            Query = f"""
            SELECT DISTINCT d.*, t.*
            FROM daily_prices d
            LEFT JOIN technical_factors t ON d.date = t.date AND t.symbol = d.symbol
            WHERE t.symbol IN ('{tickers}')
            AND d.symbol IN ('{tickers}')
            AND DATE(d.date) >= '{self.start_date}'
            AND DATE(d.date) <= '{self.end_date}'
            ORDER BY d.date
            """
        #######################################

        # Run the query and return the data
        db = sqlite3.connect(path)
        df = pd.read_sql_query(Query, db)
        self.logger.info(f"Data loaded from {path}, extract ticker: {self.tickers}")
        db.close()
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns

        # Convert the symbol column to integer type
        df["symbol"] = df["symbol"].astype("category").cat.codes
        #######################################
        df["return"] = df["close"].groupby(df["symbol"]).pct_change()
        df.fillna(0, inplace=True)
        df["log_return"] = df["return"].apply(lambda x: np.log(1 + x))

        # Add a sequential index for each symbol
        df["date_seq"] = df.groupby("symbol")["date"].cumcount()
        date = df["date"].unique()
        df = df.drop(["date"], axis=1)
        #######################################
        return df, date

    def take_step(self):
        obs = (
            self.data[self.data["date_seq"] == self.step]
            .drop(["date_seq", "symbol"], axis=1)
            .values.reshape(self.tickers_num, -1)
        )
        market_return = np.array(
            self.data[self.data["date_seq"] == self.step]["return"].values
        )
        self.step += 1
        return obs, market_return

    def reset(self):
        self.step = 0


def make_env(config: dict, logger):
    if config["strategy"]["environment"] == "single":
        data_source = DataSorce_single(config, logger)
    elif config["strategy"]["environment"] == "portfolio":
        data_source = DataSorce_portfolio(config, logger)
    else:
        raise ValueError(f"Unknown env type {config['strategy']['environment']}")

    return data_source
