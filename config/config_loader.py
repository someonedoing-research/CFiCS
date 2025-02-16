# config_loader.py

import yaml
from typing import Any, Dict

def load_config(config_path: str = "config/config.yml") -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Example validation
    required_sections = ['data', 'model', 'training', 'cross_validation']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing '{section}' section in configuration.")
    
    # Validate WANDB settings if enabled
    if config['training'].get('wandb', {}).get('enabled', False):
        wandb_required = ['project']#, 'entity']
        for param in wandb_required:
            if param not in config['training']['wandb']:
                raise ValueError(f"Missing '{param}' in 'training.wandb' configuration.")
    
    # Ensure float types for critical parameters
    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    
    if config['cross_validation'].get('wandb', {}).get('enabled', False):
        wandb_required = ['project']#, 'entity']
        for param in wandb_required:
            if param not in config['cross_validation']['wandb']:
                raise ValueError(f"Missing '{param}' in 'cross_validation.wandb' configuration.")
    
    return config