import os


def get_directory_names(path_core_data, dirs_names):
	dir_dic = {dir_name: f"{path_core_data}/{dir_name}" for dir_name in dirs_names}
	return dir_dic


def ensure_dir(folder):
    """Ensure that a directory exists. If it doesn't, create it."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        pass