import os


def get_directory_names(path_core_data, dirs_names):
    """Create a dictionary mapping directory names to their full paths.
    
    Args:
        path_core_data (str): Base path for all directories
        dirs_names (list): List of directory names
        
    Returns:
        dict: Dictionary mapping directory names to their full paths
    """
    dir_dic = {dir_name: f"{path_core_data}/{dir_name}" for dir_name in dirs_names}
    return dir_dic


def ensure_dir(folder):
    """Ensure that a directory exists. If it doesn't, create it.
    
    Args:
        folder (str): Path to the directory to ensure exists
    """
    if not os.path.exists(folder):
        os.makedirs(folder)