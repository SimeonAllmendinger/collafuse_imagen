import sys
import os
sys.path.append(os.path.abspath(os.curdir))

from src.components.nodes.base_node import BaseNode

class Cloud(BaseNode):
    def __init__(self, 
                 device,
                 model_type: str,
                 dataset_name: str,
                 idx: int,
                 path_model_save_dir: str
                 ):
        
        node_type='Cloud'
        super().__init__(_id=f"CLOUD_{idx}", 
                         node_type=node_type, 
                         device=device,
                         path_model_save_dir=path_model_save_dir,
                         model_type=model_type, 
                         dataset_name=dataset_name)