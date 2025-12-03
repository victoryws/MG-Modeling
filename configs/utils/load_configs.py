import os
import glob
from omegaconf import OmegaConf


def load_all_yaml_to_dict(yaml_folder_path):
    if not os.path.isdir(yaml_folder_path):
        raise FileNotFoundError(f"The specified path {yaml_folder_path} is not a valid folder.")
    
    yaml_files = glob.glob(os.path.join(yaml_folder_path, "*.yaml"))
    yaml_files.extend(glob.glob(os.path.join(yaml_folder_path, "*.yml")))

    config_dict = {}

    for yaml_file in yaml_files:
        file_name = os.path.basename(yaml_file)
        config_name = file_name.replace('.yaml', '').replace('.yml', '')
        
        config = OmegaConf.load(yaml_file)
        config_dict[config_name] = config

    return config_dict