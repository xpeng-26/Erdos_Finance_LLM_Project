from .manager.stock_data import StockDataManager

def ingest_stock_data(config: dict, logger) -> None:
    """Entry point for stock data ingestion"""
    manager = StockDataManager(config, logger)
    manager.run() 