from .engine.factor_calculator import FactorCalculator


def calculate_factors(config: dict, logger) -> None:
    """Calculate technical factors using the provided price data"""
    # Initialize factor calculator
    factor_calculator = FactorCalculator(config, logger)
    
    # Update factors using the update task
    factor_calculator.run()