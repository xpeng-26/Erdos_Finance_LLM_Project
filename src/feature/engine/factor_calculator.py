import pandas as pd
import numpy as np
import talib
import sqlite3
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
from utils.database.manager import DatabaseManager
from utils.database.schema import create_schema


class FactorCalculator:
    """Class for calculating technical analysis factors"""

    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize the FactorCalculator with config and logger"""
        self.config = config
        self.logger = logger
        self.factors = list(config["feature"]["factor_parameters"].keys())
        self.factor_params = config["feature"]["factor_parameters"]

        # Initialize database manager
        self.db_path = (
            Path(config["info"]["local_data_path"])
            / "data_raw"
            / config["info"]["db_name"]
        )
        self.db_manager = DatabaseManager(self.db_path)

        # path to the factor update record file
        self.update_json = (
            Path(self.config["info"]["local_data_path"])
            / "data_raw"
            / "stock_data_factor_records.json"
        )

        # delete the previous record file and create a new one
        self._create_new_record()

        # Initialize factor calculation functions and validate periods
        self._init_factor_functions()
        self._validate_config_periods()

    def _get_max_timeperiod(self) -> int:
        """Get the maximum timeperiod from factor parameters"""

        max_period = 0
        for factor, params in self.factor_params.items():
            period = params.get("timeperiod", 0)
            max_period = max(max_period, period)
        return max_period

    def _validate_config_periods(self) -> None:
        """Validate that the configured data period is sufficient for all timeperiods"""
        try:
            start_date = datetime.strptime(
                self.config["info"]["start_date"], "%Y-%m-%d"
            )
            end_date = datetime.strptime(self.config["info"]["end_date"], "%Y-%m-%d")
            period_days = (end_date - start_date).days

            if period_days < self._get_max_timeperiod():
                raise ValueError(
                    f"Config error: data period ({period_days} days) is insufficient as the maximum factor timeperiod is {self._get_max_timeperiod()} days"
                )

        except Exception as e:
            self.logger.error(f"Error validating config periods: {str(e)}")
            raise

    def _init_factor_functions(self) -> None:
        """Initialize the technical analysis functions based on config"""
        self.factor_functions = {
            # momentum
            "rsi_6": lambda x: talib.RSI(
                x["close"], timeperiod=self.factor_params["rsi_6"]["timeperiod"]
            ),
            "rsi_12": lambda x: talib.RSI(
                x["close"], timeperiod=self.factor_params["rsi_12"]["timeperiod"]
            ),
            "rsi_24": lambda x: talib.RSI(
                x["close"], timeperiod=self.factor_params["rsi_24"]["timeperiod"]
            ),
            "roc_14": lambda x: talib.ROC(
                x["close"], timeperiod=self.factor_params["roc_14"]["timeperiod"]
            ),
            "roc_30": lambda x: talib.ROC(
                x["close"], timeperiod=self.factor_params["roc_30"]["timeperiod"]
            ),
            "roc_60": lambda x: talib.ROC(
                x["close"], timeperiod=self.factor_params["roc_60"]["timeperiod"]
            ),
            "mom_14": lambda x: talib.MOM(
                x["close"], timeperiod=self.factor_params["mom_14"]["timeperiod"]
            ),
            "mom_30": lambda x: talib.MOM(
                x["close"], timeperiod=self.factor_params["mom_30"]["timeperiod"]
            ),
            "mom_60": lambda x: talib.MOM(
                x["close"], timeperiod=self.factor_params["mom_60"]["timeperiod"]
            ),
            # trend
            "ma_20": lambda x: talib.SMA(
                x["close"], timeperiod=self.factor_params["ma_20"]["timeperiod"]
            ),
            "ma_30": lambda x: talib.SMA(
                x["close"], timeperiod=self.factor_params["ma_30"]["timeperiod"]
            ),
            "ma_60": lambda x: talib.SMA(
                x["close"], timeperiod=self.factor_params["ma_60"]["timeperiod"]
            ),
            "ma_200": lambda x: talib.SMA(
                x["close"], timeperiod=self.factor_params["ma_200"]["timeperiod"]
            ),
            "ema_20": lambda x: talib.EMA(
                x["close"], timeperiod=self.factor_params["ema_20"]["timeperiod"]
            ),
            "ema_30": lambda x: talib.EMA(
                x["close"], timeperiod=self.factor_params["ema_30"]["timeperiod"]
            ),
            "ema_60": lambda x: talib.EMA(
                x["close"], timeperiod=self.factor_params["ema_60"]["timeperiod"]
            ),
            "ema_200": lambda x: talib.EMA(
                x["close"], timeperiod=self.factor_params["ema_200"]["timeperiod"]
            ),
            "macd": lambda x: talib.MACD(
                x["close"],
                fastperiod=self.factor_params["macd"]["fastperiod"],
                slowperiod=self.factor_params["macd"]["slowperiod"],
                signalperiod=self.factor_params["macd"]["signalperiod"],
            ),
            "adx": lambda x: talib.ADX(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.factor_params["adx"]["timeperiod"],
            ),
            # volatility
            "bbands": lambda x: talib.BBANDS(
                x["close"],
                timeperiod=self.factor_params["bbands"]["timeperiod"],
                nbdevup=self.factor_params["bbands"]["nbdevup"],
                nbdevdn=self.factor_params["bbands"]["nbdevdn"],
            ),
            "cci": lambda x: talib.CCI(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.factor_params["cci"]["timeperiod"],
            ),
            "atr": lambda x: talib.ATR(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.factor_params["atr"]["timeperiod"],
            ),
            # volume
            "obv": lambda x: talib.OBV(x["close"], x["volume"]),
            "ad": lambda x: talib.AD(x["high"], x["low"], x["close"], x["volume"]),
        }
        for factor in self.factors:
            if factor not in self.factor_functions:
                self.logger.warning(f"Factor {factor} not implemented yet")

    def calculate_factors(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical factors for the given price data"""
        if price_df.empty:
            return pd.DataFrame()

        factor_df = pd.DataFrame(index=price_df.index)

        for factor, func in self.factor_functions.items():
            try:
                if factor == "macd":
                    factor_macd = func(price_df)
                    factor_df[factor.lower() + "_fast"] = factor_macd[0]
                    factor_df[factor.lower() + "_slow"] = factor_macd[1]
                    factor_df[factor.lower() + "_signal"] = factor_macd[2]
                elif factor == "bbands":
                    factor_bbands = func(price_df)
                    factor_df[factor.lower() + "_upper"] = factor_bbands[0]
                    factor_df[factor.lower() + "_middle"] = factor_bbands[1]
                    factor_df[factor.lower() + "_lower"] = factor_bbands[2]
                else:
                    factor_df[factor.lower()] = func(price_df)
            except Exception as e:
                self.logger.error(f"Error calculating {factor}: {str(e)}")

        return factor_df

    def _validate_factor_names(self) -> bool:
        """Check if all configured factors exist in the factor table"""
        try:
            existing_columns = self.db_manager.query(
                "SELECT * FROM technical_factors LIMIT 1"
            ).columns
            existing_factors = {
                col.lower() for col in existing_columns if col not in ["date", "symbol"]
            }
            config_factors = {f.lower() for f in self.factors}
            if "macd" in config_factors:
                config_factors.remove("macd")
                config_factors.add("macd_fast")
                config_factors.add("macd_slow")
                config_factors.add("macd_signal")
            if "bbands" in config_factors:
                config_factors.remove("bbands")
                config_factors.add("bbands_upper")
                config_factors.add("bbands_middle")
                config_factors.add("bbands_lower")
            missing_factors = config_factors - existing_factors
            if missing_factors:
                raise ValueError(f"Missing factors in database: {missing_factors}")
            return True

        except Exception as e:
            self.logger.error(f"Error validating factor names: {str(e)}")
            raise

    def _create_new_record(self) -> None:
        """Create a new record file"""
        if Path(self.update_json).exists():
            self.logger.info(
                f"Factor update record file {self.update_json} exists, deleting it"
            )
            # delete the original file
            Path(self.update_json).unlink()
        # create a new record file
        with open(self.update_json, "w") as f:
            json.dump({}, f, indent=4)
        self.logger.info(f"New factor update record file {self.update_json} created")

    def _update_record(self, ticket: str) -> None:
        """Load and validate update task from JSON file"""
        try:
            # load the stock data download record file
            price_record_file = (
                Path(self.config["info"]["local_data_path"])
                / "data_raw"
                / "stock_data_download_records.json"
            )
            with open(price_record_file, "r") as f:
                price_record = json.load(f)
            # select the record for the current ticket
            current_record = {ticket: price_record[ticket]}
            current_record[ticket]["last_update"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # read the current record file
            with open(self.update_json, "r") as f:
                records = json.load(f)

            # append the current record dictionary to the record file
            records[ticket] = current_record[ticket]
            with open(self.update_json, "w") as f:
                json.dump(records, f, indent=4)
            self.logger.info(f"Factor update record file {self.update_json} created")
            return

        except Exception as e:
            self.logger.error(f"Error taking record: {str(e)}")
            raise

    def update_factors(self, price_df: pd.DataFrame, ticket: str) -> None:
        """Main update method with validation steps"""
        try:
            # calculate the factors
            factor_df = self.calculate_factors(price_df)
            self.logger.info(f"Factor calculation completed for {ticket}")

            # add the ticket and date to the factor dataframe
            factor_df["symbol"] = ticket
            factor_df["date"] = price_df["date"]

            # insert the factors into the database
            self.db_manager.insert(factor_df, "technical_factors")

            # update the record file
            self._update_record(ticket)

        except Exception as e:
            self.logger.error(f"Error in update_factors: {str(e)}")
            raise

    def run(self) -> None:
        """Main execution method to process all tickets and update their factors"""

        try:
            # Check if the table exists, delete it and create a new one
            if self.db_manager.check_table_exists("technical_factors"):
                self.logger.info("Technical factors table exists, deleting it")
                self.db_manager.delete_table("technical_factors")

            self.db_manager.setup_table(create_schema("technical_factors"))
            self.logger.info("Technical factors table created")

            # Validate the factor names
            self._validate_factor_names()

            # Get price table from database
            price_table = self.db_manager.query(
                "SELECT * FROM daily_prices ORDER BY symbol, date"
            )

            if price_table.empty:
                self.logger.error("No price data found in database")
                return None

            # Convert date column to datetime index with UTC timezone
            # and remove the timezone information
            # price_table['date'] = pd.to_datetime(price_table['date'], utc=True)
            # price_table.set_index('date', inplace=True)

            # Get unique tickets
            tickets = price_table["symbol"].unique()
            total_tickets = len(tickets)

            self.logger.info(f"Starting factor calculation for {total_tickets} tickets")

            # Process each ticket
            for idx, ticket in enumerate(tickets, 1):
                try:
                    self.logger.info(
                        f"Processing ticket {ticket} ({idx}/{total_tickets})"
                    )

                    # Get price data for current ticket
                    ticket_data = price_table[price_table["symbol"] == ticket].copy()
                    # sort by date in ascending order
                    # ticket_data.sort_values(by='date', ascending=True, inplace=True) not needed as already sorted by symbol and date

                    if ticket_data.empty:
                        self.logger.warning(f"No price data found for ticket {ticket}")
                        continue

                    # Update factors for the ticket
                    self.update_factors(ticket_data, ticket)

                except Exception as e:
                    self.logger.error(f"Error processing ticket {ticket}: {str(e)}")
                    continue

            self.logger.info("Completed factor calculation for all tickets")

        except Exception as e:
            self.logger.error(f"Error in run method: {str(e)}")
            raise
