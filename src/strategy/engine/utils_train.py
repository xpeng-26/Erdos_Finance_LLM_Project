import os
import torch
import pandas as pd
import joblib
from datetime import datetime



####################################################################################################
# Define the time format function
def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

####################################################################################################
# Keep only the 10 most recent model and result files
def keep_recent_files(directory, pattern, max_files, logger):
    """
    Removes older files in a directory that match a specific pattern, keeping only the most recent ones.

    Args:
        directory (str): The path to the directory containing the files.
        pattern (str): A substring pattern to match filenames.
        max_files (int): The maximum number of recent files to keep.
        logger (logging.Logger): A logger instance to log the deletion of old files.

    Behavior:
        - The function identifies all files in the specified directory that contain the given pattern.
        - It sorts the files by their last modification time in descending order (newest first).
        - If the number of matching files exceeds `max_files`, the oldest files are deleted.
        - Logs the deletion of each file using the provided logger.

    Raises:
        FileNotFoundError: If the directory does not exist.
        PermissionError: If the program lacks permissions to access or delete files.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if pattern in f]
    files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time (newest first)
    if len(files) > max_files:
        for file_to_delete in files[max_files:]:
            os.remove(file_to_delete)
            #logger.info(f"Deleted old file: {file_to_delete}")

####################################################################################################
# Dump the latest DDQN model using joblib
def dump_latest_DDQN_model(trading_agent, model_path, logger, env, news):
    """
    Save the latest DDQN model using joblib.

    Args:
        trading_agent (DDQNAgent): The trading agent instance.
        model_path (str): Path to save the model.
        logger (Logger): Logger instance for logging.
        env (str): Environment name.

    Returns:
        str: Path to the saved model.
    """
    # Ensure the model directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model with a timestamped name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path_timestamped = os.path.join(model_path, f'trading_agent_{env}_{news}_model_{timestamp}.joblib')
    joblib.dump(trading_agent.online_model, model_path_timestamped)
    logger.info(f"Model saved: {model_path_timestamped}")




####################################################################################################
# Define the load checkpoint function
def load_checkpoint(trading_agent, model_path, device, logger, env, news):
    """
    Load the latest checkpoint if it exists.

    Args:
        trading_agent (DDQNAgent): The trading agent instance.
        model_path (str): Path to the directory containing the checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        int: The episode to resume training from, or 1 if no checkpoint exists.
    """
    checkpoint_path = os.path.join(model_path, f'trading_agent_{env}_{news}_checkpoint_latest.pt')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model and optimizer states
        trading_agent.online_model.load_state_dict(checkpoint['model_state_dict'])
        trading_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore other states
        trading_agent.epsilon = checkpoint['epsilon']
        trading_agent.losses = checkpoint['losses']
        episode = checkpoint['episode']

        logger.info(f"Checkpoint loaded. Resuming from episode {episode}.")
        return episode
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        return 1
    
def load_final_model(trading_agent, model_path, device, logger, env, news):
    """
    Load the latest final model with a timestamp if it exists.

    Args:
        trading_agent (DDQNAgent): The trading agent instance.
        model_path (str): Path to the directory containing the final model.
        device (torch.device): The device to load the model onto.
        logger (Logger): Logger instance for logging.
        env (str): Environment name.

    Returns:
        bool: True if the model was successfully loaded, False otherwise.
    """
    # Find all files matching the pattern for the final model
    pattern = f'final_model_{env}_{news}_*.pth'
    model_files = [f for f in os.listdir(model_path) if f.startswith(f'final_model_{env}_{news}_') and f.endswith('.pth')]

    if not model_files:
        logger.info("No final model found.")
        return False

    # Sort files by modification time (newest first)
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)

    # Get the latest model file
    latest_model_file = model_files[0]
    final_model_path = os.path.join(model_path, latest_model_file)

    # Load the model
    logger.info(f"Loading final model from {final_model_path}...")
    trading_agent.online_model.load_state_dict(torch.load(final_model_path, map_location=device))
    logger.info("Final model loaded successfully.")
    return True
    
# Define the save checkpoint function
def save_checkpoint(trading_agent, episode, config, model_path, navs, market_navs, diffs, logger, env, news):
    """
    Save the model, optimizer, and training results as a checkpoint.

    Args:
        trading_agent (DDQNAgent): The trading agent instance.
        episode (int): The current episode number.
        config (dict): The configuration dictionary.
        model_path (str): Path to save the model checkpoint.
        navs (list): List of agent NAVs.
        market_navs (list): List of market NAVs.
        diffs (list): List of differences between agent and market NAVs.
        logger (Logger): Logger instance for logging.
    """
    # Ensure the model directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save model, optimizer, and other states
    checkpoint = {
        'model_state_dict': trading_agent.online_model.state_dict(),
        'optimizer_state_dict': trading_agent.optimizer.state_dict(),
        'epsilon': trading_agent.epsilon,
        'episode': episode,
        'config': config,
        'losses': trading_agent.losses
    }

    # Save with a timestamped name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Save the checkpoint with a timestamp
    checkpoint_path_timestamped = os.path.join(model_path, f'trading_agent_{env}_{news}_checkpoint_{timestamp}.pt')
    torch.save(checkpoint, checkpoint_path_timestamped)
    logger.info(f"Checkpoint saved: {checkpoint_path_timestamped}")

    # Save with a fixed name for the latest checkpoint
    checkpoint_path_latest = os.path.join(model_path, f'trading_agent_{env}_{news}_checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_path_latest)
    logger.info(f"Latest checkpoint saved: {checkpoint_path_latest}")

    # Ensure all lists are the same length
    min_length = min(len(navs), len(market_navs), len(diffs))
    navs = navs[:min_length]
    market_navs = market_navs[:min_length]
    diffs = diffs[:min_length]
    episodes = list(range(1, min_length + 1))

    # Save the results
    results = pd.DataFrame({
        'Episode': episodes,
        'Agent': navs,
        'Market': market_navs,
        'Difference': diffs
    }).set_index('Episode')

    # Add rolling strategy win percentage
    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()

    # Save results to CSV
    result_path = os.path.join(config['info']['local_data_path'], 'evaluation')
    os.makedirs(result_path, exist_ok=True)
    results_path = os.path.join(result_path, f'training_results_{env}_{news}_{timestamp}.csv')
    results.to_csv(results_path)
    logger.info(f"Training results saved: {results_path}")

    # Keep only the 10 most recent files
    keep_recent_files(result_path, f'training_results_{env}_{news}', 10, logger)
    keep_recent_files(model_path, f'trading_agent_{env}_{news}_checkpoint', 10, logger)


