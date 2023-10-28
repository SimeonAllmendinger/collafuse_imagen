import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
from functools import partial

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms as T
from torchvision.datasets.mnist import MNIST, FashionMNIST

from src.components.utils.settings import Settings
from src.components.utils import functions as func
from src.components.nodes.base_node import BaseNode

SETTINGS=Settings()

def tmp_func(x):
        return (x - 0.5) * 2
class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder

        maybe_convert_fn = partial(func.convert_image_to_fn, convert_image_to) if func.exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Lambda(lambda x: (x - 0.5) * 2),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def tmp_func(x):
        return (x - 0.5) * 2
    
class Client(BaseNode):
    def __init__(self, 
                 idx: int,
                 device, 
                 dataset_function,
                 image_chw: int
                 ):
        
        id=f'CLIENT_{idx}'
        node_type='Client'
        super().__init__(id, node_type, device=device)
        
        # Data Handling
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(tmp_func),
            T.Resize(image_chw[1:],antialias=True),
            T.CenterCrop(image_chw[1]),
        ])
        #self.transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: (x - 0.5) * 2)], T.Resize(image_chw[1:]),)
        self.ds_train=dataset_function("./data", download=True, train=True, transform=self.transform)
        self.ds_test=dataset_function("./data", download=True, train=False, transform=self.transform)
        
        if torch.cuda.is_available():
            SETTINGS.logger.info(f'Current device name of {self.id}: {torch.cuda.get_device_name(device=device)}')
        
    def set_dl(self, batch_size) -> DataLoader:
        # Data Loader
        self.dl_train=DataLoader(self.ds_train, batch_size, shuffle=True, num_workers=0)
        self.dl_test=DataLoader(self.ds_test, batch_size, shuffle=True, num_workers=0)
