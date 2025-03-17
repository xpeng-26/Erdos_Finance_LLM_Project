from .manager.stock_data import StockDataManager
from .manager.factor_calculator import FactorCalculator

def ingest_stock_data(config: dict, logger) -> None:
    """Entry point for stock data ingestion"""
    # Initialize managers
    stock_manager = StockDataManager(config, logger)
    factor_manager = FactorCalculator(config, logger)
    
    # Run stock data ingestion
    stock_manager.run()
    
    # Update technical factors
    update_technical_factors(stock_manager, factor_manager, logger)

def update_technical_factors(stock_manager: StockDataManager, factor_manager: FactorCalculator, logger) -> None:
    """
    Update technical factors after stock data ingestion
    
    Args:
        stock_manager (StockDataManager): Stock data manager instance
        factor_manager (FactorCalculator): Factor calculator instance
        logger: Logger instance
    """
    try:
        # Get latest price data
        price_df = stock_manager.get_latest_price_data()
        if price_df is None or price_df.empty:
            logger.warning("No price data available for factor calculation")
            return
            
        # Get existing factor data if any
        existing_factor_df = stock_manager.get_existing_factor_data()
        
        # Calculate/update factors
        updated_factor_df = factor_manager.update_factors(price_df, existing_factor_df)
        
        # Store updated factors
        if not updated_factor_df.empty:
            stock_manager.store_factor_data(updated_factor_df)
            logger.info("Technical factors updated successfully")
        else:
            logger.warning("No factors were calculated or updated")
            
    except Exception as e:
        logger.error(f"Error updating technical factors: {str(e)}")
        raise 