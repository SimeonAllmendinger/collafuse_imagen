import os
import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
from functools import partial

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.mnist import MNIST, FashionMNIST

from torchvision import transforms as T

from src.components.utils.settings import Settings
from src.components.utils import functions as func
from src.components.nodes.base_node import BaseNode

SETTINGS=Settings()
LOGGER=SETTINGS.logger()

class Dataset(Dataset):
    def __init__(
        self,
        folder: str,
        image_chw: int,
        data_sample_interval: list, # interval of data sample
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.data_sample_min, self.data_sample_max = data_sample_interval

        maybe_convert_fn = partial(func.convert_image_to_fn, convert_image_to) if func.exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(maybe_convert_fn),
            T.Lambda(func.normalize_to_neg_one_to_one),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.Resize(image_chw[1:],antialias=True),
            T.CenterCrop(image_chw[1])
        ])
        
        self.image_chw = image_chw
        self.paths = [p for ext in exts for k in range(self.data_sample_min, self.data_sample_max) for p in Path(f'{folder}').glob(f'**/{k:03d}-*.{ext}')]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        label=0
        return self.transform(img), label
    
class Client(BaseNode):
    def __init__(self, 
                idx: int,
                device, 
                dataset_name: str,
                image_chw: int,
                data_train_sample_interval: list,
                data_test_sample_interval: list,
                t_cut_ratio: float,
                path_tmp_dir: str
                ):
        
        # Call the parent class constructor
        super().__init__(id=f'CLIENT_{idx}', node_type='Client', device=device)
        
        # Set the t_cut_ratio attribute
        self.t_cut_ratio = t_cut_ratio
        
        # Set the transform attribute based on the dataset_name
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Lambda(lambda x: (x - 0.5) * 2),
            T.Resize(image_chw[1:])
        ])
        
        # Initialize the datasets based on the dataset_name
        if dataset_name == 'MNIST':
            self.ds_train = MNIST(f"{path_tmp_dir}/data", download=True, train=True, transform=self.transform)
            self.ds_test = MNIST(f"{path_tmp_dir}/data", download=True, train=False, transform=self.transform)
        elif dataset_name == 'FashionMNIST':
            self.ds_train = FashionMNIST(f"{path_tmp_dir}/data", download=True, train=True, transform=self.transform)
            self.ds_test = FashionMNIST(f"{path_tmp_dir}/data", download=True, train=False, transform=self.transform)
        elif dataset_name == 'BraTS2020':
            self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_train_sliced']), 
                                    image_chw=image_chw,
                                    data_sample_interval=data_train_sample_interval)
            self.ds_test = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_test_sliced']), 
                                image_chw=image_chw,
                                data_sample_interval=data_test_sample_interval)
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
        
        # Log the length of the train dataset
        LOGGER.debug(f'Train Dataset length: {len(self.ds_train)}')
        
        # Log the current device name if CUDA is available
        if torch.cuda.is_available():
            LOGGER.info(f'Current device name of {self.id}: {torch.cuda.get_device_name(device=device)}')
        
    def set_dl(self, batch_size: int, num_workers: int) -> DataLoader:
        """
        Set the data loaders for training and testing.

        Args:
            batch_size (int): The batch size for the data loader.
            num_workers (int): The number of workers for the data loader.

        Returns:
            DataLoader: The data loader for training.
        """
        # Create the data loader for training
        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Create the data loader for testing
        self.dl_test = DataLoader(
            self.ds_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Return the data loader for training
        return self.dl_train
