from .engine.stock_data import StockDataManager
from .engine.news_data import NewsDataDownloader

def ingest_stock_data(config: dict, logger) -> None:
    """Entry point for stock data ingestion"""
    # Initialize managers
    stock_manager = StockDataManager(config, logger)

    # Run stock data ingestion
    stock_manager.run()

def ingest_news_data(config: dict, logger) -> None:
    """Function to ingest news data"""
    # Initialize NewsDataDownloader
    news_downloader = NewsDataDownloader(config, logger)

    # Download news data
    news_data = news_downloader.run()
