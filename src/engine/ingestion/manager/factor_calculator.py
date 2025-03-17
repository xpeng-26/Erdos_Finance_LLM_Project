import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Optional
import logging
from datetime import datetime

class FactorCalculator:
    """Class for calculating technical analysis factors"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """
        Initialize the FactorCalculator
        
        Args:
            config (dict): Configuration dictionary containing factor settings
            logger (logging.Logger): Logger instance
        """
        self.config = config
        self.logger = logger
        self.factors = config.get('factors', [])
        self.factor_params = config.get('factor_parameters', {})
        
        # Initialize factor calculation functions
        self._init_factor_functions()
    
    def _init_factor_functions(self) -> None:
        """Initialize the technical analysis functions based on config"""
        self.factor_functions = {}
        for factor in self.factors:
            if factor == 'RSI':
                self.factor_functions[factor] = lambda x: talib.RSI(x, timeperiod=self.factor_params.get('RSI', {}).get('timeperiod', 14))
            elif factor == 'ROC':
                self.factor_functions[factor] = lambda x: talib.ROC(x, timeperiod=self.factor_params.get('ROC', {}).get('timeperiod', 10))
            elif factor == 'MOM':
                self.factor_functions[factor] = lambda x: talib.MOM(x, timeperiod=self.factor_params.get('MOM', {}).get('timeperiod', 10))
            else:
                self.logger.warning(f"Factor {factor} not implemented yet")
    
    def validate_dates(self, price_df: pd.DataFrame, factor_df: Optional[pd.DataFrame] = None) -> tuple[bool, pd.DataFrame]:
        """
        Validate and synchronize dates between price and factor data
        
        Args:
            price_df (pd.DataFrame): DataFrame containing price data
            factor_df (Optional[pd.DataFrame]): DataFrame containing existing factor data
            
        Returns:
            tuple[bool, pd.DataFrame]: (is_full_recalc_needed, dates_to_calculate)
        """
        if factor_df is None or factor_df.empty:
            return True, price_df
            
        # Get date ranges
        price_dates = set(price_df.index)
        factor_dates = set(factor_df.index)
        
        # Check for date mismatches
        if price_dates != factor_dates:
            self.logger.info("Date mismatch detected. Full recalculation needed.")
            return True, price_df
            
        # Check if we need to add new dates
        new_dates = price_dates - factor_dates
        if new_dates:
            self.logger.info(f"New dates detected: {len(new_dates)}")
            return False, price_df.loc[new_dates]
            
        return False, pd.DataFrame()
    
    def calculate_factors(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical factors for the given price data
        
        Args:
            price_df (pd.DataFrame): DataFrame containing price data
            
        Returns:
            pd.DataFrame: DataFrame containing calculated factors
        """
        if price_df.empty:
            return pd.DataFrame()
            
        close_prices = price_df['close']
        factor_df = pd.DataFrame(index=price_df.index)
        
        for factor, func in self.factor_functions.items():
            try:
                factor_df[f"{factor.lower()}"] = func(close_prices)
            except Exception as e:
                self.logger.error(f"Error calculating {factor}: {str(e)}")
                
        return factor_df
    
    def update_factors(self, price_df: pd.DataFrame, existing_factor_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Update technical factors based on price data changes
        
        Args:
            price_df (pd.DataFrame): DataFrame containing price data
            existing_factor_df (Optional[pd.DataFrame]): DataFrame containing existing factor data
            
        Returns:
            pd.DataFrame: Updated factor DataFrame
        """
        is_full_recalc, dates_to_calculate = self.validate_dates(price_df, existing_factor_df)
        
        if is_full_recalc:
            self.logger.info("Performing full factor recalculation")
            return self.calculate_factors(price_df)
        else:
            self.logger.info("Calculating factors for new dates only")
            new_factors = self.calculate_factors(dates_to_calculate)
            if existing_factor_df is not None:
                return pd.concat([existing_factor_df, new_factors])
            return new_factors 