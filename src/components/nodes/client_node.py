import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from pathlib import Path
from functools import partial
from cleanfid import fid, features, clip_features
from cleanfid import utils as fid_utils

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
        t_cut_ratio: float,
        client_id: str,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        is_dataset_results=False,
        is_dataset_cloud=False
        ):
        
        super().__init__()
        self.folder = folder
        if data_sample_interval:
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
        if is_dataset_results:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'{int(t_cut_ratio*100)}/image_{client_id}_*.{ext}')]
        elif is_dataset_cloud:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'{int(t_cut_ratio*100)}_cloud/image_{client_id}_*.{ext}')]
        else:
            self.paths = [p for ext in exts for k in range(self.data_sample_min, self.data_sample_max) for p in Path(f'{folder}').glob(f'**/{k:03d}-*.{ext}')]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label=0
        
        img = Image.open(path)
        img= self.transform(img)
        
        if self.image_chw[0] == 3:
            # create new image of desired size and color (blue) for padding
            new_image_width = 299
            new_image_height = 299
            result = np.zeros((new_image_height,new_image_width), dtype=np.uint8)

            # compute center offset
            x_center = (new_image_width - self.image_chw[2]) // 2
            y_center = (new_image_height - self.image_chw[1]) // 2

            # copy img image into center of result image
            result[y_center:y_center+self.image_chw[1], x_center:x_center+self.image_chw[2]] = func.unnormalize_to_zero_to_one(img.cpu().numpy().squeeze()* 255).astype(np.uint8)
            result=np.stack((result,)*3, axis=-1)
            
            img=torch.tensor(func.normalize_to_neg_one_to_one(result)).permute(2,0,1)
        
        return img, label
    
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
                                    image_chw=(3,128,128),#image_chw,
                                    data_sample_interval=data_train_sample_interval,
                                    t_cut_ratio=self.t_cut_ratio,
                                    client_id=self.id)
            self.ds_test = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_test_sliced']), 
                                image_chw=(3,128,128),
                                data_sample_interval=data_test_sample_interval,
                                t_cut_ratio=self.t_cut_ratio,
                                client_id=self.id)
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
        
        # Log the length of the train dataset
        LOGGER.debug(f'Train Dataset length: {len(self.ds_train)}')
        
        # Log the current device name if CUDA is available
        if torch.cuda.is_available():
            LOGGER.info(f"Current device name of {self.id}: {device} | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else 'cpu')
    
    @property
    def path_save_model(self):
        path = self.diffusion_model.path_save_model.replace('.pt',f'_{self.t_cut_ratio}.pt')
        return path
        
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

    def compute_performance(self):
        
        #* kid
        kid_feature_model = features.build_feature_extractor(mode="clean", device=torch.device("cuda:0"))
        kid_features_actual, kid_features_synthetic, kid_features_test_data = self.get_performance_features(feature_model=kid_feature_model)

        self.kid_score_train = fid.kernel_distance(feats1=kid_features_actual,feats2=kid_features_synthetic,num_subsets=10, max_subset_size=400)
        self.kid_score_test = fid.kernel_distance(feats1=kid_features_test_data,feats2=kid_features_synthetic,num_subsets=10, max_subset_size=400)
        
        LOGGER.info(f'CLEAN_FID_SCORE_TRAIN-{self.id}: {self.kid_score_train}')
        LOGGER.info(f'CLEAN_FID_SCORE_TEST-{self.id}: {self.kid_score_test}')
        
        #* fid
        #fid_feature_model = features.build_feature_extractor(mode="legacy_pytorch", device=self.device)
        
        #* fcd
        #fcd_feature_model = clip_features.CLIP_fx("ViT-B/32", device=self.device)

    def compute_information_disclosure(self):
        #* kid
        kid_feature_model = features.build_feature_extractor(mode="clean", device=torch.device("cuda:0"))
        kid_features_actual, kid_features_synthetic, kid_features_test_data = self.get_inf_dis_features(feature_model=kid_feature_model)

        self.kid_inf_dis_train = fid.kernel_distance(feats1=kid_features_actual,feats2=kid_features_synthetic,num_subsets=10, max_subset_size=400)
        self.kid_inf_dis_test = fid.kernel_distance(feats1=kid_features_test_data,feats2=kid_features_synthetic,num_subsets=10, max_subset_size=400)
        
    @torch.inference_mode()
    def get_performance_features(self, feature_model):
        
        # Data
        if not self.dl_test:
            self.set_dl(batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'], 
                        num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
            
        ds_results = Dataset(folder=os.path.join(SETTINGS.diffusion_trainer['DEFAULT']['results_folder'],'testing'),
                             image_chw=(3,128,128),
                             data_sample_interval=None,
                             is_dataset_results=True,
                             t_cut_ratio=self.t_cut_ratio,
                             client_id=self.id)
        
        dl_results = DataLoader(ds_results,
                                batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'],
                                shuffle=True,
                                num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
        
        feature_list_actual = []
        feature_list_synthetic = []
        feature_list_test_data = []
        
        for batch in self.dl_train:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_actual.append(features)
            
        for batch in self.dl_test:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_test_data.append(features)
        
        for batch in dl_results:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_synthetic.append(features)
        
        features_actual = np.concatenate(feature_list_actual)
        features_synthetic = np.concatenate(feature_list_synthetic)
        features_test_data = np.concatenate(feature_list_test_data)
        
        return features_actual, features_synthetic, features_test_data

    @torch.inference_mode()
    def get_inf_dis_features(self, feature_model):
        
        # Data
        if not self.dl_test:
            self.set_dl(batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'], 
                        num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
            
        ds_inf_dis = Dataset(folder=os.path.join(SETTINGS.diffusion_trainer['DEFAULT']['results_folder'],'testing'),
                             image_chw=(3,128,128),
                             data_sample_interval=None,
                             is_dataset_cloud=True,
                             t_cut_ratio=self.t_cut_ratio,
                             client_id=self.id)
        
        dl_inf_dis = DataLoader(ds_inf_dis,
                                batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'],
                                shuffle=True,
                                num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
        
        feature_list_actual = []
        feature_list_inf_dis = []
        feature_list_test_data = []
        
        for batch in self.dl_train:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_actual.append(features)
            
        for batch in self.dl_test:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_test_data.append(features)
        
        for batch in dl_inf_dis:
            features = feature_model(batch[0].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_inf_dis.append(features)
        
        features_actual = np.concatenate(feature_list_actual)
        features_inf_dis = np.concatenate(feature_list_inf_dis)
        features_test_data = np.concatenate(feature_list_test_data)
        
        return features_actual, features_inf_dis, features_test_data      