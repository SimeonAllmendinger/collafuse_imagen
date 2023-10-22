from src.components.utils.settings import Settings
from src.components.model.ddpm import DDPM
from src.components.model.unet import UNet

SETTINGS=Settings()

class BaseNode():
    def __init__(self, id: str, node_type: str, device: str):
        # Properties
        self._id=id
        self.node_type=node_type
        
        # Model
        self.ddpm=DDPM(network=UNet(**SETTINGS.unet['DEFAULT']), device=device, **SETTINGS.ddpm['DEFAULT'])
        SETTINGS.logger.info('Number of parameters:', sum([p.numel() for p in self.ddpm.parameters()]))
    
    @property
    def id(self):
        return self._id
	
    @id.setter
    def id(self, id: str):
        SETTINGS.logger.warning(f"Id of node type {self.node_type} is changed from {self.id} to {id}")
        self._id=id

    @classmethod
    def create_collaborator_node(self, device: str):
        id='COLLABORATOR_1'
        node_type='Collaborator'
        return self(id, node_type, device)
    
    @classmethod
    def create_cloud_node(self, device: str):
        id='CLOUD_1'
        node_type='Cloud'
        return self(id=id, node_type=node_type, device=device)