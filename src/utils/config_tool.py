import tomllib
import tomli_w
import os

def parse_config(config_path):
		"""
		Parse the toml configuration file
		"""
		with open(config_path, "rb") as f:
				config = tomllib.load(f)
		return config

def save_config_copy(config_path, config, file_name):
		"""
		Save a copy of the configuration file
		"""
		file_name = file_name.split(".")[0] + config['date']['today'] + ".toml"
		with open(os.path.join(config_path, file_name), "wb") as f:
				tomli_w.dump(config, f)
		return None