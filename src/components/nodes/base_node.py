from codecarbon import EmissionsTracker
from codecarbon.output import EmissionsData
from collections import defaultdict

from src.components.utils.settings import Settings
from src.components.model.diffusion import Diffusion_Model
from src.components.model.unet import Unet

SETTINGS=Settings()
    
class BaseNode():
    def __init__(self, id: str, node_type: str, device: str):
        # Properties
        self._id=id
        self.node_type=node_type
        
        # Model
        self.diffusion_model=Diffusion_Model(unet=Unet(**SETTINGS.unet['DEFAULT']), device=device, path_save_model=f'./src/assets/diffusion_model_{self.id}.pt', **SETTINGS.diffusion_model['DEFAULT'])
        SETTINGS.logger.info('Number of parameters:', sum([p.numel() for p in self.diffusion_model.parameters()]))
        
        # Resources
        self.tracker = EmissionsTracker()
        self.energy_usage = defaultdict(EmissionsData)
    
    @property
    def id(self):
        return self._id
	
    @id.setter
    def id(self, id: str):
        SETTINGS.logger.warning(f"Id of node type {self.node_type} is changed from {self.id} to {id}")
        self._id=id