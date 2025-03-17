from .manager.stock_data import StockDataManager

def ingest_stock_data(config: dict, logger) -> None:
    """Entry point for stock data ingestion"""
    # Initialize managers
    stock_manager = StockDataManager(config, logger)

    # Run stock data ingestion
    stock_manager.run()
