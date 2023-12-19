import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

from src.components.utils.logger import _get_logger_

import os

def get_main_working_directory(name):
    path_base = os.getcwd()
    
    while path_base and not path_base.endswith(name):
        path_base = os.path.dirname(path_base)
    
    assert path_base, 'Could not find current directory'
    
    return path_base

class Settings():
    
    path_cwd=get_main_working_directory('collafuse')
    __base = {'PATH_BASE_DIR': path_cwd}
    __logger = _get_logger_(path_base_dir=__base['PATH_BASE_DIR'], verbose=False) # .setLevel(logging.INFO)
    
    def __init__(self):
        
        # Utils
        self.__logger.debug(f'BASE_PATH: {self.__base["PATH_BASE_DIR"]}')
        
        # Models
        self.unet = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/model/config_unet.yaml'))
        self.diffusion_model = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/model/config_diffusion_model.yaml'))
        self.diffusion_trainer = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/model/config_diffusion_trainer.yaml'))
        self.imagen_model = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/model/config_imagen_model.yaml'))
        self.efficient_unet = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/model/config_efficient_unet.yaml'))
        
        # Nodes
        self.clients = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/nodes/config_clients.yaml'))
        self.clouds = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/nodes/config_clouds.yaml'))
        
        # Data
        self.data = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'configs/utils/config_data.yaml'))
        
        # Visualization tools
        #self.wandb = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/visualization/config_wandb.yaml'))
        
    @staticmethod
    def logger():
        """Function to get logger

        Returns:
            logging.Logger: logger object
        """    
        return Settings.__logger

def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)
        
    return config

def main():
    SETTINGS = Settings()
    SETTINGS.logger.info(SETTINGS.diffusion_model)

if __name__ == '__main__':
    main()