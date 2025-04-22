import os
import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque
from pathlib import Path
from utils.database.manager import DatabaseManager
from utils.database.schema import create_schema


class NewsDataDownloader:
    def __init__(self, config, logger):
        self.logger = logger
        self.overwrite_news_table = config["ingestion"]["overwrite_news_table"]

        # Load environment variables from config/confidential.env
        env_path = Path("config/confidential.env")
        load_dotenv(dotenv_path=env_path)

        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found in environment variables. Check that ALPHA_VANTAGE_API_KEY is set in config/confidential.env"
            )
        self.start_datetime = (
            config["ingestion"]["start_date"] + " 00:00:00"
        )  # YYYY-MM-DD HH:MM:SS
        self.end_datetime = (
            config["ingestion"]["end_date"] + " 23:59:59"
        )  # YYYY-MM-DD HH:MM:SS
        # Get the news limitation per api call from config
        self.news_limit_per_api_call = config["ingestion"].get(
            "news_limit_per_api_call", 100
        )
        # Get the API call total limit from config
        self.api_call_total_limit = config["ingestion"].get("api_call_total_limit", 5)
        # Get the API calls per minute limit from config
        self.api_calls_per_minute_limit = config["ingestion"].get(
            "api_calls_per_minute_limit", 5
        )
        # Initialize API call counter
        self.api_call_count = 0
        # Use a deque to efficiently track recent API calls (timestamps within the last minute)
        self.recent_api_calls = deque()

        # Initialize database manager
        self.raw_data_path = Path(config["info"]["local_data_path"]) / "data_raw"
        self.db_path = self.raw_data_path / config["info"]["db_news_name"]
        self.db_manager = DatabaseManager(self.db_path)

        # Set up stats file path
        self.raw_data_path = Path(config["info"]["local_data_path"]) / "data_raw"
        self.stats_file = self.raw_data_path / "news_stats.json"

        # read the stock_symbol_list file
        stock_symbol_list_df = pd.read_csv(
            self.raw_data_path / config["ingestion"]["stock_symbol_list"]
        )
        self.symbols = stock_symbol_list_df["Symbol"].tolist()

    def create_news_stats(self):
        """
        Create a JSON file with news statistics for each ticker by querying the database.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            stats = {}
            # delete the json file if it exists
            if self.stats_file.exists():
                self.stats_file.unlink()

            for symbol in self.symbols:
                # Skip if symbol is not in database
                check_query = f"SELECT COUNT(*) FROM news WHERE symbol = '{symbol}'"
                check_df = self.db_manager.query(check_query)
                if check_df.empty or check_df.iloc[0, 0] == 0:
                    self.logger.info(f"No news data found for {symbol}")
                    continue

                # Get earliest and latest dates
                date_query = f"""
                SELECT 
                    MIN(datetime) as earliest_date, 
                    MAX(datetime) as latest_date, 
                    COUNT(*) as news_count 
                FROM news 
                WHERE symbol = '{symbol}'
                """
                stats_df = self.db_manager.query(date_query)

                # Format dates nicely for output
                earliest_date = stats_df.iloc[0]["earliest_date"]
                if (
                    earliest_date
                    and isinstance(earliest_date, str)
                    and "T" in earliest_date
                ):
                    earliest_date = earliest_date.split("T")[0]

                latest_date = stats_df.iloc[0]["latest_date"]
                if latest_date and isinstance(latest_date, str) and "T" in latest_date:
                    latest_date = latest_date.split("T")[0]

                # Store statistics for this symbol
                stats[symbol] = {
                    "earliest_news_date": earliest_date,
                    "latest_news_date": latest_date,
                    "news_count": int(stats_df.iloc[0]["news_count"]),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

            # Add summary stats
            total_query = "SELECT COUNT(*) as total FROM news"
            total_df = self.db_manager.query(total_query)
            total_news = int(total_df.iloc[0]["total"]) if not total_df.empty else 0

            stats["_summary"] = {
                "total_news_count": total_news,
                "tickers_with_news": len(stats),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Write to file with nice formatting
            with open(self.stats_file, "w") as f:
                json.dump(stats, f, indent=4)

            self.logger.info(f"Created news statistics file at {self.stats_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating news statistics: {str(e)}")
            return False

    def _is_rate_limited(self):
        """
        Check if we've exceeded the API calls per minute limit.

        Returns:
            bool: True if rate limited, False otherwise
        """
        # Calculate the cutoff time (1 minute ago)
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)

        # Remove timestamps older than 1 minute
        while self.recent_api_calls and self.recent_api_calls[0] < one_minute_ago:
            self.recent_api_calls.popleft()

        # Check if we've made too many calls in the last minute
        return len(self.recent_api_calls) >= self.api_calls_per_minute_limit

    def _make_api_call(self, url, params):
        """
        Make an API call with rate limiting.

        Args:
            url (str): The API URL
            params (dict): The API parameters

        Returns:
            dict: The API response JSON
        """
        # Wait until we're not per-minute rate limited anymore
        while self._is_rate_limited() and self.recent_api_calls:
            self.logger.warning(
                f"API rate limit of {self.api_calls_per_minute_limit} calls per minute reached."
            )
            oldest_timestamp = self.recent_api_calls[0]
            time_diff = (datetime.now() - oldest_timestamp).seconds
            wait_time = max(1, 60 - time_diff + 1)  # Ensure at least 1 second wait
            self.logger.info(f"Waiting {wait_time} seconds before next API call")
            time.sleep(wait_time)
            # _is_rate_limited will automatically clean up expired timestamps

        # Check if we've reached the API call limit
        if self.api_call_count >= self.api_call_total_limit:
            self.logger.warning(
                f"API call total limit reached ({self.api_call_total_limit}). Cannot make more calls."
            )
            return None

        # Increment API call counter and record timestamp
        self.api_call_count += 1
        current_time = datetime.now()
        self.recent_api_calls.append(current_time)
        self.logger.info(
            f"Making API call {self.api_call_count}/{self.api_call_total_limit}"
        )

        # Add retry logic for connection issues
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params)
                return response.json()
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
            ) as e:
                retry_count += 1
                self.logger.warning(
                    f"Connection error: {str(e)}. Retry {retry_count}/{max_retries}"
                )
                if retry_count < max_retries:
                    # Wait a bit before retrying (exponential backoff)
                    time.sleep(2**retry_count)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} retries")
                    raise

    def download_ticker_news(self, symbol):
        """
        Download news for a single ticker symbol.

        Args:
            symbol (str): The ticker symbol to download news for

        Returns:
            pandas.DataFrame: A DataFrame of news items for the given ticker
        """
        ticker_news = []
        batch_start_datetime = datetime.strptime(
            self.start_datetime, "%Y-%m-%d %H:%M:%S"
        )
        config_end_datetime = datetime.strptime(self.end_datetime, "%Y-%m-%d %H:%M:%S")

        self.logger.info(
            f"Downloading news for {symbol} from {batch_start_datetime} to {config_end_datetime}"
        )

        # download news by batch, 1000 max news items per batch (one api call)

        # continue if the batch_start_date_time is 1 minute before the config_end_date_time
        while (config_end_datetime - batch_start_datetime).total_seconds() >= 60:

            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.api_key,
                "time_from": batch_start_datetime.strftime("%Y%m%dT%H%M"),
                "time_to": config_end_datetime.strftime("%Y%m%dT%H%M"),
                "limit": self.news_limit_per_api_call,
                "sort": "EARLIEST",
            }

            data = self._make_api_call(url, params)
            if data is None:  # API call limit reached
                break

            # No news to fetch, done for this ticker
            if "feed" not in data or len(data["feed"]) == 0:
                self.logger.warning(
                    f"No news found for {symbol} in the date range {batch_start_datetime} to {config_end_datetime}"
                )
                break

            # Process the news items
            batch_size = int(data["items"] if "items" in data else len(data["feed"]))
            batch_news = []

            for item in data["feed"]:
                news = {
                    "symbol": symbol,
                    "datetime": datetime.strptime(
                        item["time_published"], "%Y%m%dT%H%M%S"
                    ),
                    "title": item["title"],
                    "source": item["source"],
                    "summary": item["summary"],
                    "length_summary": len(item["summary"]),
                    "url": item["url"],
                    "overall_sentiment": item["overall_sentiment_label"],
                    "overall_sentiment_score": item["overall_sentiment_score"],
                    "created_at": datetime.now(),  # Current timestamp for when the record is created
                }

                # Add ticker-specific sentiment if available
                if "ticker_sentiment" in item:
                    for ticker_data in item["ticker_sentiment"]:
                        if ticker_data["ticker"] == symbol:
                            news["sentiment"] = ticker_data["ticker_sentiment_label"]
                            news["sentiment_score"] = ticker_data[
                                "ticker_sentiment_score"
                            ]
                            news["relevance_score"] = ticker_data["relevance_score"]
                            break

                batch_news.append(news)

            ticker_news.extend(batch_news)

            # If we got fewer items than the limit, we've done for this ticker
            if batch_size < int(self.news_limit_per_api_call):
                self.logger.info(
                    f"Retrieved {batch_size} news items (fewer than limit). No more pagination needed."
                )
                break

            # If reached the limitation, need to paginate by adjusting the date range
            latest_datetime = max(news["datetime"] for news in batch_news)
            # Update batch_start_datetime to be one minute after the latest date
            batch_start_datetime = str(latest_datetime + timedelta(minutes=1))
            batch_start_datetime = datetime.strptime(
                batch_start_datetime, "%Y-%m-%d %H:%M:%S"
            )
            self.logger.info(
                f"Adjusting date range to {batch_start_datetime} to {config_end_datetime} for next batch"
            )

        # get the datetime range from ticker_news
        if ticker_news:
            earliest_datetime = min(news["datetime"] for news in ticker_news)
            latest_datetime = max(news["datetime"] for news in ticker_news)
            self.logger.info(
                f"Downloaded a total of {len(ticker_news)} news items for {symbol} from {earliest_datetime} to {latest_datetime}"
            )
            return pd.DataFrame(ticker_news)
        else:
            self.logger.warning(
                f"No news data downloaded for {symbol} in the entire date range"
            )
            return pd.DataFrame()

    def store_news_in_db(self, news_df, symbol):
        """
        Store news data in the database.

        Args:
            news_df (pandas.DataFrame): DataFrame containing news data to store
            symbol (str): The ticker symbol

        Returns:
            int: Number of news items stored
        """
        if news_df.empty:
            return

        # sort the news data by date in ascending order
        news_df = news_df.sort_values(by="datetime", ascending=True)
        try:
            self.db_manager.insert(news_df, "news")
            news_count = len(news_df)
            self.logger.info(f"Stored {news_count} news items in database for {symbol}")
            return
        except Exception as e:
            self.logger.error(f"Error storing news in database: {str(e)}")
            raise

    def download_news(self):
        """
        Download news for all symbols and store in the database.

        Returns:
            pandas.DataFrame: A DataFrame containing all downloaded news
        """
        n_news_data = 0
        for symbol in self.symbols:
            # Check if we've reached the API call limit
            if self.api_call_count >= self.api_call_total_limit:
                self.logger.warning(
                    f"API call total limit reached ({self.api_call_total_limit}). Stopping news download."
                )
                break

            # Get news for this ticker
            ticker_news_df = self.download_ticker_news(symbol)

            # Store the ticker news in the database
            if not ticker_news_df.empty:
                self.store_news_in_db(ticker_news_df, symbol)
                n_news_data += len(ticker_news_df)

        # Log the total count
        if n_news_data > 0:
            self.logger.info(
                f"Downloaded and stored a total of {n_news_data} news items with {self.api_call_count} API calls"
            )
            return
        else:
            self.logger.info(
                f"No news data downloaded with {self.api_call_count} API calls"
            )
            return

    def run(self):
        """
        Download news data, store in database, and create statistics file.
        """
        try:
            # Check if the news table exists and if it does, delete it if overwrite_news_table is True
            if self.db_manager.check_table_exists("news") and self.overwrite_news_table:
                self.db_manager.delete_table("news")
                self.logger.info("News table deleted in the database")
            # Create the news table if it doesn't exist
            self.db_manager.setup_table(create_schema("news"))
            self.logger.info("News table created in the database")

            # Download and store news data
            self.download_news()

            # Create statistics file
            self.logger.info("Creating news statistics file...")
            self.create_news_stats()

            return

        except Exception as e:
            self.logger.error(f"Error in news ingestion process: {str(e)}")
            raise
