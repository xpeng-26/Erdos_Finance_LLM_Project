import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple, List, Dict
from ..update_task import UpdateTask
from utils.database.manager import DatabaseManager
from utils.database.schema import create_schema
import pandas as pd

class StockDataManager:
    """Manager class for stock data operations"""
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.raw_data_path = Path(config['info']['local_data_path']) / 'data_raw'
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.record_file = self.raw_data_path / 'stock_data_download_records.json'
        self.db_path = self.raw_data_path / config['info']['db_name']
        
        # Initialize database managers for price and factor data
        self.price_db_manager = DatabaseManager(self.db_path, create_schema('daily_prices'))
        self.factor_db_manager = DatabaseManager(self.db_path, create_schema('technical_factors'))

    def validate_dates(self) -> bool:
        """Validate configuration dates"""
        try:
            start = datetime.strptime(self.config['info']['start_date'], '%Y-%m-%d')
            end = datetime.strptime(self.config['info']['end_date'], '%Y-%m-%d')
            if start >= end:
                raise ValueError(f"Start date ({start}) must be before end date ({end})")
            return True
        except ValueError as e:
            self.logger.error(f"Date validation error: {str(e)}")
            return False

    def get_symbol_dates(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Get start and end dates for symbol from records
        
        Args:
            symbol (str): Stock symbol to look up
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (start_date, end_date) from records,
                or (None, None) if no records exist
        """
        if not self.record_file.exists():
            with open(self.record_file, 'w') as f:
                json.dump({}, f, indent=4)
                
        with open(self.record_file, 'r') as f:
            try:
                records = json.load(f)
                if symbol in records:
                    return (records[symbol]['last_start_date'], 
                           records[symbol]['last_end_date'])
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON in {self.record_file}, creating new records")
                with open(self.record_file, 'w') as f:
                    json.dump({}, f, indent=4)
        return None, None

    def determine_updates(self, symbol: str, record_start: Optional[str], 
                         record_end: Optional[str]) -> List[UpdateTask]:
        """Determine required updates for a symbol"""
        updates: List[UpdateTask] = []
        config_start = self.config['info']['start_date']
        config_end = self.config['info']['end_date']

        # Check for backward update
        if record_start and datetime.strptime(config_start, '%Y-%m-%d') < datetime.strptime(record_start, '%Y-%m-%d'):
            # Calculate end date as one day before record_start
            end_date = (datetime.strptime(record_start, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            updates.append(UpdateTask(
                start=config_start,
                end=end_date,
                direction='backward'
            ))
            self.logger.info(f"Will update {symbol} backward: {config_start} to {end_date}")

        # Check for forward update or full download
        if not record_end:
            updates.append(UpdateTask(config_start, config_end, 'full'))
            self.logger.info(f"Will download full history for {symbol}")
        elif datetime.strptime(config_end, '%Y-%m-%d') > datetime.strptime(record_end, '%Y-%m-%d'):
            next_date = (datetime.strptime(record_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            updates.append(UpdateTask(next_date, config_end, 'forward'))
            self.logger.info(f"Will update {symbol} forward: {next_date} to {config_end}")

        return updates

    def update_record_file(self, symbol: str, update: UpdateTask):
        """Update the JSON record file for a symbol
        
        Args:
            symbol (str): Stock symbol
            update (UpdateTask): The update task that was completed
        """
        # Ensure the file exists with initial content
        if not self.record_file.exists():
            with open(self.record_file, 'w') as f:
                json.dump({}, f, indent=4)
        
        try:
            # Read existing records
            with open(self.record_file, 'r') as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in {self.record_file}, creating new records")
                    records = {}
            
            # Update the record for this symbol
            if symbol not in records:
                records[symbol] = {}
            
            # Update dates using existing records if available
            current_record = records[symbol]
            records[symbol] = {
                'last_start_date': min(
                    update.start,
                    current_record.get('last_start_date', update.start)
                ),
                'last_end_date': max(
                    update.end,
                    current_record.get('last_end_date', update.end)
                ),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Write all records back to file
            with open(self.record_file, 'w') as f:
                json.dump(records, f, indent=4)
            
            self.logger.debug(f"Updated records for {symbol}: {records[symbol]}")
            
        except Exception as e:
            self.logger.error(f"Error updating record file for {symbol}: {str(e)}")
            raise

    def process_price_data(self, symbol: str, update: UpdateTask) -> Optional[pd.DataFrame]:
        """Process price data for a given symbol and update task
        
        Args:
            symbol (str): Stock symbol
            update (UpdateTask): The update task containing date range
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing price data or None if error
        """
        try:
            # Get price data
            df = yf.Ticker(symbol).history(start=update.start, end=update.end)
            
            if df.empty:
                self.logger.info(f"No price data available for {symbol} in {update.direction} update")
                return None

            # Process and prepare data for storage
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            df_to_store = df[['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df_to_store.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            
            # Validate price data before storing
            if df_to_store.isnull().any().any():
                self.logger.warning(f"Found missing values in price data for {symbol}, skipping")
                return None
                
            return df_to_store
            
        except Exception as e:
            self.logger.error(f"Error processing price data for {symbol}: {str(e)}")
            return None

    def process_symbol(self, symbol: str):
        """Process updates for a single symbol"""
        record_start, record_end = self.get_symbol_dates(symbol)
        updates = self.determine_updates(symbol, record_start, record_end)

        if not updates:
            self.logger.info(f"Data for {symbol} is already up to date")
            return

        for update in updates:
            try:
                self.logger.info(f"Processing {update.direction} update for {symbol}")
                
                # Get and store price data
                price_data = self.process_price_data(symbol, update)
                if price_data is None:
                    self.logger.warning(f"No price data available for {symbol}, skipping update")
                    continue
                
                # Store price data
                self.price_db_manager.insert(price_data)
                self.logger.info(f"Successfully stored price data for {symbol}")
                
                # Update record file
                self.update_record_file(symbol, update)

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")

    def run(self):
        """Main execution method"""
        if not self.validate_dates():
            return

        try:
            # Setup database manager
            self.price_db_manager.setup()
            
            # Process each symbol
            for symbol in self.config['ingestion']['stock_list']:
                self.process_symbol(symbol)
        finally:
            # Close database connection
            self.price_db_manager.close()

    def get_latest_price_data(self) -> Optional[pd.DataFrame]:
        """
        Get the latest price data from the database
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing price data or None if error
        """
        try:
            query = """
                SELECT date, symbol, open, high, low, close, volume
                FROM daily_prices
                ORDER BY date DESC
            """
            df = self.price_db_manager.execute_query(query)
            if df is not None and not df.empty:
                df.set_index('date', inplace=True)
                return df
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price data: {str(e)}")
            return None

    def get_existing_factor_data(self) -> Optional[pd.DataFrame]:
        """
        Get existing factor data from the database
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing factor data or None if error
        """
        try:
            query = """
                SELECT date, symbol, rsi_14, roc_10, mom_10
                FROM technical_factors
                ORDER BY date DESC
            """
            df = self.factor_db_manager.execute_query(query)
            if df is not None and not df.empty:
                df.set_index('date', inplace=True)
                return df
            return None
        except Exception as e:
            self.logger.error(f"Error getting existing factor data: {str(e)}")
            return None

    def store_factor_data(self, factor_df: pd.DataFrame) -> bool:
        """
        Store factor data in the database
        
        Args:
            factor_df (pd.DataFrame): DataFrame containing factor data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if factor_df.empty:
                self.logger.warning("No factor data to store")
                return False
                
            # Reset index to make date a column
            factor_df = factor_df.reset_index()
            
            # Prepare data for storage
            factor_df['date'] = pd.to_datetime(factor_df['date']).dt.strftime('%Y-%m-%d')
            
            # Store in database
            self.factor_db_manager.insert_dataframe(factor_df, 'technical_factors')
            self.logger.info(f"Successfully stored factor data for {len(factor_df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing factor data: {str(e)}")
            return False 