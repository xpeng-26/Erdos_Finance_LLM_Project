from .engine.factor_calculator import FactorCalculator
from .engine.llm_analyst import LlmAnalyst


def calculate_factors(config: dict, logger) -> None:
    """Calculate technical factors using the provided price data"""
    # Initialize factor calculator
    factor_calculator = FactorCalculator(config, logger)

    # Update factors using the update task
    factor_calculator.run()


def inference_ai_sentiment_advisory(config: dict, logger) -> None:
    """Inference AI sentiment advisory using the provided news data"""
    # Initialize AI sentiment advisory
    ai_sentiment_advisory = LlmAnalyst(config, logger)

    # Inference AI sentiment advisory
    ai_sentiment_advisory.run()

