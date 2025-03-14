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

# utils
from utils.config_tool import parse_config, save_config_copy
from utils.directory_tool import ensure_dir, get_directory_names
from utils.logging_tool import initialize_logger

# custom modules
from engine.data.stock_driver import ingest_stock_data

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
		config['date']['today'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


		# Data directories
		# Forming data directories in the local data path
		dirs = get_directory_names(
			path_core_data = config["info"]["local_data_path"],
			dirs_names = config["info"]["dirs_names"]
		)


		# Save the config as a copy
		ensure_dir(dirs["config_archive"])
		save_config_copy(
			config_path = dirs["config_archive"],
			config = config,
			file_name = "trade_w_llm_copy_.toml"
		)


		# Logging
		# Initialize the logger
		ensure_dir(dirs["logs"])
		log_file = "{}_log_{}.txt".format(
			os.path.splitext(os.path.basename(config_filename))[0],
			config["date"]['today'])
		logger = initialize_logger(
			log_path = dirs["logs"],
			log_file = log_file
		)



		############################################
		# Starting pipeline

		# Ingest stock data
		if config['pipeline']['ingestion']:
			logger.info('Start ingesting stock data...')
			ensure_dir(dirs["data_raw"])

			# Call the ingestion function with the config and logger
			ingest_stock_data(config, logger)

			logger.info('Stock data ingestion completed.')




############################################
if __name__ == '__main__':
		# Argument parsing
		parser = argparse.ArgumentParser(description='Trading with LLM.')
		parser.add_argument(
			'--config',
			type=str,
			default='config/trade_w_llm.toml',
			dest='config_filename',
			help='the path to the configuration file'
		)

		# Parse the arguments
		args = parser.parse_args()

		# Run the main function
		main(args)
