import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

from src.components.utils.logger import _get_logger_
    
class Settings():
    
    def __init__(self):
        #
        path_cwd=get_main_working_directory('distributedgenai')
        
        #
        self.base = {'PATH_BASE_DIR': path_cwd}
        
        # Utils
        self.logger = _get_logger_(path_base_dir=self.base['PATH_BASE_DIR'], verbose=False) # .setLevel(logging.INFO)
        self.logger.debug(f'Master_Logger started')
        self.logger.debug(f'BASE_PATH: {self.base["PATH_BASE_DIR"]}')
        
        # Models
        self.ddpm = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/model/config_ddpm.yaml'))
        self.unet = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/model/config_unet.yaml'))
        self.ddpm_trainer = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/model/config_ddpm_trainer.yaml'))
        
        # Visualization tools
        self.wandb = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/visualization/config_wandb.yaml'))
        

def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)
        
    return config


def get_main_working_directory(name):
    
    path_base = os.getcwd()
    
    for i in range(len(path_base.split('/'))):

        if path_base.split('/')[-1] == name:
            break
        else:
            path_base = '/'.join(path_base.split('/')[0:-1])
    
    assert len(path_base) > 0, 'Could not find current directory'
    
    return path_base


def main():
    SETTINGS = Settings()
    SETTINGS.logger.info(SETTINGS.imagen)

if __name__ == '__main__':
    main()