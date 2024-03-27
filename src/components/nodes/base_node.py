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
    def __init__(self, _id: str, node_type: str, device: str, model_type: str, path_model_save_dir: str, dataset_name=None):
        # Properties
        self._id=_id
        self.node_type=node_type
        self.model_type=model_type
        imagen_timesteps=SETTINGS.imagen_model['DEFAULT']['timesteps']
        
        if not dataset_name:
            dataset_name=''
        
        self.dataset_name=dataset_name
        
        # Model
        match self.model_type:
            case 'DDPM':
                self.model=Diffusion_Model(unet=Unet(**SETTINGS.unet['DEFAULT']), device=device, path_save_model=f'./src/assets/diffusion_model_{self.id}.pt', **SETTINGS.diffusion_model['DEFAULT'])
            case 'IMAGEN':
                path_save_model=os.path.join(path_model_save_dir,f"{imagen_timesteps}/imagen_model_{dataset_name}_{self.id}_s-{SETTINGS.imagen_model['DEFAULT']['image_sizes'][0]}.pt")
                self.model=Imagen(unets=[EfficientUnet(**SETTINGS.efficient_unet['UNET_64'])], path_save_model=path_save_model, **SETTINGS.imagen_model['DEFAULT']).cuda()
        
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