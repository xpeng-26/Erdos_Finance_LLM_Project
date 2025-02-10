#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the main function of all drivers to preidct stock with news.
Should use config/predict_stock_w_news.toml to configure the parameters.
"""

import os
import logging
import argparse
import errno
import datetime

from utils.config_tool import parse_config, save_config_copy
from utils.directory_tool import ensure_dir
from utils.logging_tool import initialize_logger
from utils.startup import get_directory_names

from engine.ingestion.ingest_stock_driver import ingest_stock


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

		dir_current_script = os.path.dirname(os.path.abspath(__file__))
		dir_data = os.path.abspath(os.path.join(dir_current_script, "..", "data/searching"))

		# Get the directory names
		dirs = get_directory_names(path_core_data = dir_data)
		if not dir_data:
			dir_data = dirs["data/searching"]
		
		# Import the configuration file
		config_file = os.path.join(dirs["config"], config_filename)
		if os.path.exists(config_file):
			config = parse_config(config_file)
		else:
			# Raise an error if the configuration file does not exist
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
		
		# Add today to the configuration
		config["date"]['today'] = datetime.datetime.now().strftime('%Y-%m-%d')

		# Initialize the logger
		log_file = "{}_log_{}.txt".format(
			os.path.splitext(os.path.basename(config_filename))[0],
			config["date"]['today'])
		logger = logging.getLogger(__name__)


		# Forming data directories
		path_core_data = config["info"]["gdrive_path"]

		dirs = get_directory_names(
			dirs_dict = dirs,
			dirs_names = config["info"]["dirs_names"],
			path_core_data = path_core_data,
			separater_key = ""
		)

		# Save the config as a copy
		save_config_copy(config_file, dir_data, config["date"]["today"])


		############################################
		# Starting pipeline

		# Ingest stock data
		if config['pipeline']['stock_ingestion']:
			logger.info('Start ingesting stock data...')
			ensure_dir(dirs["stock_ingestion"])

			ingest_stock(args.config)

			logger.info('Stock data ingestion completed.')


############################################
if __name__ == '__main__':
		# Argument parsing
		parser = argparse.ArgumentParser(description='Predict stock with news.')
		parser.add_argument(
			'--config',
			type=str,
			default='config/predict_stock_w_news.toml',
			dest='config_filename',
			help='the path to the configuration file'
		)

		# Parse the arguments
		args = parser.parse_args()

		# Run the main function
		main(args)
