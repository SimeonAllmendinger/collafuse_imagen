import sys
import os
sys.path.append(os.path.abspath(os.curdir))

from src.components.nodes.base_node import BaseNode

class Cloud(BaseNode):
    def __init__(self, 
                 device,
                 model_type: str,
                 dataset_name: str,
                 id: str
                 ):
        
        node_type='Cloud'
        super().__init__(id, node_type, device=device, model_type=model_type, dataset_name=dataset_name)