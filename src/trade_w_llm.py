#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the main function of all drivers to preidct stock with news.
Should use config/predict_stock_w_news.toml to configure the parameters.
"""

import os
import argparse
import errno
import datetime
import warnings

# utils
from utils.config_tool import parse_config, save_config_copy
from utils.directory_tool import ensure_dir, get_directory_names
from utils.logging_tool import initialize_logger

# custom modules
from ingestion.stock_driver import ingest_stock_data, ingest_news_data
from feature.feature_engineer_driver import (
    calculate_factors,
    inference_ai_sentiment_advisory,
)

from strategy.train_trading_agent import train_trading_agent, dump_final_DDQN
from strategy.train_PPO_agent import train_PPO_agent, train_A2C_agent
from evaluation.evaluation import evaluation_main


############################################
def main(opt_params):
    """
    The main function to predict stock with news.

    Args:
        opt_params: Optional parameters via argparse

    Returns:
        None
    """

    # Optional parameters
    config_filename = opt_params.config_filename
    dir_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Configuration
    # Import the configuration file
    config_file = os.path.join(dir_project, config_filename)
    if os.path.exists(config_file):
        config = parse_config(config_file)
    else:
        # Raise an error if the configuration file does not exist
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    # Add today to the configuration
    config["date"]["today"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Data directories
    # Forming data directories in the local data path
    dirs = get_directory_names(
        path_core_data=config["info"]["local_data_path"],
        dirs_names=config["info"]["dirs_names"],
    )

    # Save the config as a copy
    ensure_dir(dirs["config_archive"])
    save_config_copy(
        config_path=dirs["config_archive"],
        config=config,
        file_name="trade_w_llm_copy_.toml",
    )


	# Logging
    # Initialize the logger
    ensure_dir(dirs["logs"])
    log_file = "{}_log_{}.txt".format(
        os.path.splitext(os.path.basename(config_filename))[0],
        config["date"]['today'])
    logger = initialize_logger(
        log_path=dirs["logs"],
        log_file=log_file
    )

    # create all the files
    for directory in dirs.keys():
        ensure_dir(dirs[directory])

      
    ############################################
    # Starting pipeline

    if config["pipeline"]["ingestion_stock"]:
        logger.info("---------- pipeline: ingestion_stock ----------")
        ensure_dir(dirs["data_raw"])

        # Ingest stock data
        logger.info("Start ingesting stock data...")
        ingest_stock_data(config, logger)
        logger.info("Stock data ingestion completed.\n")

    if config["pipeline"]["ingestion_news"]:
        logger.info("---------- pipeline: ingestion_news ----------")
        # Ingest news data
        logger.info("Start ingesting news data...")
        ingest_news_data(config, logger)
        logger.info("News data ingestion completed.\n")

    if config["pipeline"]["feature_factor"]:
        logger.info("---------- pipeline: feature_factor ----------")

        # Calculate factors
        logger.info('Start calculating factors...')
        calculate_factors(config, logger)
        logger.info('Factors calculation completed.\n')

    if config["pipeline"]["feature_news"]:
        logger.info("---------- pipeline: feature_news ----------")

        # Calculate news features
        logger.info("Start inferencing news features (sentiment score and advisory) ...")
        inference_ai_sentiment_advisory(config, logger)
        logger.info("News features (sentiment score and advisory) inference completed.\n")

        
    if config['pipeline']['strategy']:
        logger.info('---------- pipeline: strategy ----------')
        logger.info('Start trading with reinforcement learning agent...')
        # Train which agent

    if config['strategy']['DDQN']:
        logger.info('Training with DDQN agent...')
        # Trading with reinforcement learning agent
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            train_trading_agent(config, logger)

    if config['strategy']['PPO']:
        logger.info('Training with PPO agent...')
        train_PPO_agent(config, logger)

    if config['strategy']['A2C']:
        logger.info('Training with A2C agent...')
        train_A2C_agent(config, logger)

        logger.info('Trading with reinforcement learning agent completed.\n')

    if config['pipeline']['evaluation']:
        logger.info('---------- pipeline: evaluation ----------')
        logger.info('Start evaluating the trading strategy...')
        # Dump the final model
        dump_final_DDQN(config, logger)
        logger.info('Final model dumped.\n')
        # Evaluate the trading strategy
        evaluation_main(config, logger)
        # evaluate_trading_strategy(config, logger)
        logger.info('Trading strategy evaluation completed.\n')


			



############################################
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Trading with LLM.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/trade_w_llm.toml",
        dest="config_filename",
        help="the path to the configuration file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
