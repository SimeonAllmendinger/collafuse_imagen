from codecarbon import EmissionsTracker
from codecarbon.output import EmissionsData
from collections import defaultdict

from src.components.utils.settings import Settings
from src.components.model.ddpm import Diffusion_Model
from src.components.model.imagen import Imagen
from src.components.model.efficient_unet import Unet as EfficientUnet
from src.components.model.unet import Unet

SETTINGS=Settings()
LOGGER = SETTINGS.logger()
    
class BaseNode():
    def __init__(self, id: str, node_type: str, device: str, model_type: str):
        # Properties
        self._id=id
        self.node_type=node_type
        self.model_type=model_type
        
        # Model
        match self.model_type:
            case 'DDPM':
                self.model=Diffusion_Model(unet=Unet(**SETTINGS.unet['DEFAULT']), device=device, path_save_model=f'./src/assets/diffusion_model_{self.id}.pt', **SETTINGS.diffusion_model['DEFAULT'])
            case 'IMAGEN':
                self.model=Imagen(unets=[EfficientUnet(**SETTINGS.efficient_unet['UNET_64'])], path_save_model=f'./src/assets/imagen_model_CELEB-A_{self.id}.pt', **SETTINGS.imagen_model['DEFAULT']).to(device)
        
        # Resources
        self.tracker = EmissionsTracker(save_to_logger=True, logging_logger=LOGGER, log_level="error")
        self.energy_usage = defaultdict(EmissionsData)
    
    @property
    def id(self):
        return self._id
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def image_chw(self):
        return (self.model.image_channel, self.model.image_height, self.model.image_width)
	
    @id.setter
    def id(self, id: str):
        LOGGER.warning(f"Id of node type {self.node_type} is changed from {self.id} to {id}")
        self._id=id