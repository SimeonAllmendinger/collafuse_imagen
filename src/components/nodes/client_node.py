import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial
from cleanfid import fid

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision import transforms as T

from imagen_pytorch.t5 import t5_encode_text

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
        t_cut_ratio: float,
        client_id: str,
        n_clients: int,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        path_labels=None,
        path_text_embeds=None,
        dataset_name=None,
        attributes=None,
        len_chunk=30000
        ):
        
        super().__init__()
            
        self.folder = folder
        self.dataset_name=dataset_name
        self.client_id = client_id
        self.path_text_embeds = path_text_embeds

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
        
        match self.dataset_name:
            case 'BraTS2020':
                
                data_sample_min, data_sample_max = attributes 
                
                self.paths = [p for ext in exts for k in range(data_sample_min, data_sample_max) for p in Path(f'{folder}').glob(f'**/{k:03d}-*.{ext}')]
                
            case 'CelebA':
                    
                df = pd.read_csv(path_labels, sep=r'\s+', header=1, usecols=['PATH'] + attributes)
                
                # Filter rows where at least one of the specified columns has a value of 1
                df = df[df.eq(1).any(axis=1)]
                self.paths = df['PATH'].to_list()
                
                # Extract column names with at least one '1' and convert the list of column names to a string
                self.texts = df.apply(lambda row: ', '.join(row.index[row == 1].tolist()).replace('_', ' '), axis=1).to_list()
                
                for j in tqdm(range(int(len(self.texts)/len_chunk)+1)):
                    
                    path_text_masks_file = f'{path_text_embeds}text_masks_{client_id}_{len(self.paths)}_{j}.pt'
                    path_text_embeds_file = f'{path_text_embeds}text_embeds_{client_id}_{len(self.paths)}_{j}.pt'
                    
                    if not os.path.exists(path_text_embeds_file) or not os.path.exists(path_text_masks_file):
                        if (j+1)*len_chunk > len(self.texts):
                            text_embeds, text_masks = t5_encode_text(self.texts[j*len_chunk:], 
                                                            device=torch.device(1), 
                                                            return_attn_mask = True)
                        else:
                            text_embeds, text_masks = t5_encode_text(self.texts[j*len_chunk:(j+1)*len_chunk], 
                                                            device=torch.device(1), 
                                                            return_attn_mask = True)
                        
                        # Define the number of elements to extend (zeros) on the right side
                        max_mask_size = 20
                        
                        assert text_masks.size(1) < max_mask_size, f'mask size exceeds length of {max_mask_size} with {text_masks.size(1)}'
                        
                        num_elements_to_extend = max_mask_size - text_masks.size(1)
                        
                        # Create a tensor of zeros with the desired shape
                        zeros_extension_embeds = torch.zeros((text_embeds.size(0), num_elements_to_extend, text_embeds.size(2)))
                        zeros_extension_masks = torch.zeros((text_masks.size(0), num_elements_to_extend))

                        # Concatenate the original tensor with the tensor of zeros along the second dimension
                        text_embeds = torch.cat((text_embeds.to('cpu'), zeros_extension_embeds), dim=1)
                        text_masks = torch.cat((text_masks.to('cpu'), zeros_extension_masks), dim=1) > 0

                        torch.save(text_embeds, path_text_embeds_file)
                        torch.save(text_masks, path_text_masks_file)
                    else:
                        text_embeds = torch.load(path_text_embeds_file, map_location=torch.device('cpu'))
                        text_masks = torch.load(path_text_masks_file, map_location=torch.device('cpu')) > 0
                    if j == 0:
                        self.text_embeds = text_embeds
                        self.text_masks = text_masks
                    else:
                        self.text_embeds = torch.cat((self.text_embeds, text_embeds), dim=0)
                        self.text_masks = torch.cat((self.text_masks, text_masks), dim=0)
            
            case 'Birds':
                df_img = pd.read_csv(path_labels, sep=' ', header=0)  
                df_labels = pd.read_csv('/home/woody/btr0/btr0104h/data/Birds/attributes/image_attribute_labels.txt', sep=r'\s+', header=0, usecols=[0,1,2,3])
                df_att = pd.read_csv('/home/woody/btr0/btr0104h/data/Birds/attributes/attributes.txt', sep=' ', header=0)
                
                df_labels = df_labels.loc[(df_labels.iloc[:,1].isin(attributes)) & (df_labels.iloc[:,2] == 1) & (df_labels.iloc[:,3] == 4),:]
                list_img_unique = sorted(df_labels.iloc[:,0].unique().tolist())
                
                self.paths = df_img.loc[df_img.iloc[:,0].isin(list_img_unique),:].iloc[:,1].values.tolist()
                self.texts = [', '.join([df_att.iloc[class_idx,1].replace('_',' ').replace('::', ' ') for class_idx in sorted(df_labels.loc[(df_labels.iloc[:,0] == img_idx),:].iloc[:,1].values.tolist())]) for img_idx in sorted(list_img_unique)]
                
                for path in self.paths:
                    img = Image.open(os.path.join(self.folder,path))
                    img = self.transform(img)
                    if img.shape[0] != 3:
                        index = self.paths.index(path)
                        self.texts.pop(index)
                        self.paths.pop(index)
                    
                self.len_chunk=30000
                for j in tqdm(range(int(len(self.texts)/self.len_chunk)+1)):
                    
                    path_text_masks_file = f'{path_text_embeds}text_masks_{client_id}_{len(self.paths)}_{j}.pt'
                    path_text_embeds_file = f'{path_text_embeds}text_embeds_{client_id}_{len(self.paths)}_{j}.pt'
                    
                    if not os.path.exists(path_text_embeds_file) or not os.path.exists(path_text_masks_file):
                        if (j+1)*self.len_chunk > len(self.texts):
                            text_embeds, text_masks = t5_encode_text(self.texts[j*self.len_chunk:], 
                                                            device=torch.device(1), 
                                                            return_attn_mask = True)
                        else:
                            text_embeds, text_masks = t5_encode_text(self.texts[j*self.len_chunk:(j+1)*self.len_chunk], 
                                                            device=torch.device(1), 
                                                            return_attn_mask = True)
                        
                        # Define the number of elements to extend (zeros) on the right side
                        max_mask_size = 35
                        
                        assert text_masks.size(1) <= max_mask_size, f'mask size exceeds length of {max_mask_size} with {text_masks.size(1)}'
                        
                        num_elements_to_extend = max_mask_size - text_masks.size(1)
                        
                        # Create a tensor of zeros with the desired shape
                        zeros_extension_embeds = torch.zeros((text_embeds.size(0), num_elements_to_extend, text_embeds.size(2)))
                        zeros_extension_masks = torch.zeros((text_masks.size(0), num_elements_to_extend))

                        # Concatenate the original tensor with the tensor of zeros along the second dimension
                        text_embeds = torch.cat((text_embeds.to('cpu'), zeros_extension_embeds), dim=1)
                        text_masks = torch.cat((text_masks.to('cpu'), zeros_extension_masks), dim=1) > 0

                        torch.save(text_embeds, path_text_embeds_file)
                        torch.save(text_masks, path_text_masks_file)
                    else:
                        text_embeds = torch.load(path_text_embeds_file, map_location=torch.device('cpu'))
                        text_masks = torch.load(path_text_masks_file, map_location=torch.device('cpu')) > 0
                    if j == 0:
                        self.text_embeds = text_embeds
                        self.text_masks = text_masks
                    else:
                        self.text_embeds = torch.cat((self.text_embeds, text_embeds), dim=0)
                        self.text_masks = torch.cat((self.text_masks, text_masks), dim=0)
                        
                LOGGER.debug(f'Number of images: {self.paths[:5]}')
                LOGGER.debug(f'Number of texts: {self.texts[:5]}')
                LOGGER.debug(f'Number of text embeds: {self.text_embeds.size()}')
                LOGGER.debug(f'Number of text masks: {self.text_masks.size()}')
            
            case 'STL10':
                df = pd.read_csv(path_labels, 
                                sep=' ', header=None, 
                                names=['PATH_IMG', 'CLASS', 'NAME'])
                df = df.loc[(df.CLASS >= self.data_sample_min) & (df.CLASS < self.data_sample_max),:]
            
                self.paths = df.PATH_IMG.to_list()
                self.texts = df.NAME.astype(str).to_list() 
            
                path_text_masks_file = f'{path_text_embeds}text_masks_{client_id}_{len(self.paths)}.pt'
                path_text_embeds_file = f'{path_text_embeds}text_embeds_{client_id}_{len(self.paths)}.pt'
                if not os.path.exists(path_text_embeds_file) or not os.path.exists(path_text_masks_file):
                    self.text_embeds, self.text_masks = t5_encode_text(self.texts, 
                                                    device=torch.device(1), 
                                                    return_attn_mask = True)
                    torch.save(self.text_embeds, path_text_embeds_file)
                    torch.save(self.text_masks, path_text_masks_file)
                self.text_embeds = torch.load(path_text_embeds_file, map_location=torch.device('cpu'))
                self.text_masks = torch.load(path_text_masks_file, map_location=torch.device('cpu'))
            
            case _:
                LOGGER.error(f'Unknown dataset: {dataset_name}')
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        img= self.transform(img)
        
        if self.dataset_name != 'BraTS2020':
            assert img.shape[0] == 3, f'{path} has image shape of {img.shape[0]}'  
            text = self.texts[index]
        
            match self.dataset_name:
                case 'STL10':
                    label = (self.text_embeds[index], self.text_masks[index])
                case 'CelebA':
                    label = (self.text_embeds[index], self.text_masks[index])
                case 'Birds':
                    label = (self.text_embeds[index], self.text_masks[index])
        
        if self.dataset_name == 'BraTS2020':
            label = 0
            text = 0
        
        return img, label, text
    
    
class Client(BaseNode):
    def __init__(self, 
                idx: int,
                device, 
                dataset_name: str,
                image_chw: int,
                n_train_items: int,
                n_test_items: list,
                t_cut_ratio: float,
                path_tmp_dir: str,
                model_type: str,
                n_clients: int,
                path_model_save_dir: str,
                celeb_a_attributes: [str] = None,
                birds_attributes: [str] = None,
                seed: int = 42,
                ):
        
        # Call the parent class constructor
        super().__init__(_id=f'CLIENT_{idx}', 
                         node_type='Client', 
                         device=device, 
                         model_type=model_type, 
                         path_model_save_dir=path_model_save_dir,
                         dataset_name=dataset_name)
        
        # Set the t_cut_ratio attribute
        self.t_cut_ratio = t_cut_ratio
        
        # Set the transform attribute based on the dataset_name
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Lambda(lambda x: (x - 0.5) * 2),
            T.Resize(image_chw[1:])
        ])
        
        # Initialize the datasets based on the dataset_name
        match dataset_name:
            case 'MNIST':
                self.ds_train = MNIST(f"{path_tmp_dir}/data", download=True, train=True, transform=self.transform)
                self.ds_test = MNIST(f"{path_tmp_dir}/data", download=True, train=False, transform=self.transform)
            case 'FashionMNIST':
                self.ds_train = FashionMNIST(f"{path_tmp_dir}/data", download=True, train=True, transform=self.transform)
                self.ds_test = FashionMNIST(f"{path_tmp_dir}/data", download=True, train=False, transform=self.transform)
            case 'BraTS2020':
                # Define the data sample interval for training and testing
                n_client_train_items = int(SETTINGS.data['BraTS2020']['n_train_items']/n_clients)
                n_client_test_items = int(SETTINGS.data['BraTS2020']['n_test_items']/n_clients)
                data_train_sample_interval = [int((idx-1) * n_client_train_items), int(idx * n_client_train_items)]
                data_test_sample_interval = [int((idx-1) * n_client_test_items), int(idx * n_client_test_items)]
                
                # Initialize the datasets
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_train_sliced']), 
                                        image_chw=image_chw,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id,
                                        n_clients=n_clients,
                                        dataset_name=dataset_name,
                                        attributes=data_train_sample_interval)
                self.ds_test = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_test_sliced']), 
                                       image_chw=image_chw,
                                       t_cut_ratio=self.t_cut_ratio,
                                       client_id=self.id,
                                       n_clients=n_clients,
                                       dataset_name=dataset_name,
                                       attributes=data_test_sample_interval)
            
            case 'CelebA':
                self.ds = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_train_images']), 
                                    image_chw=image_chw,
                                    t_cut_ratio=self.t_cut_ratio,
                                    client_id=self.id,
                                    path_labels=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_labels']),
                                    path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_text_embeds']),
                                    dataset_name=dataset_name,
                                    attributes=celeb_a_attributes,
                                    n_clients=n_clients)
                g_cpu = torch.Generator()
                g_cpu.manual_seed(seed)
                self.ds_train, self.ds_test, self.ds_remainder = random_split(dataset=self.ds, 
                                                                              lengths=[n_train_items, n_test_items, len(self.ds) - n_train_items - n_test_items],
                                                                              generator=g_cpu)
            
            case 'Birds':
                self.ds = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_train_images']), 
                                    image_chw=image_chw,
                                    t_cut_ratio=self.t_cut_ratio,
                                    client_id=self.id,
                                    path_labels=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_labels']),
                                    path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_text_embeds']),
                                    dataset_name=dataset_name,
                                    attributes=birds_attributes,
                                    n_clients=n_clients)
                g_cpu = torch.Generator()
                g_cpu.manual_seed(seed)
                self.ds_train, self.ds_test, self.ds_remainder = random_split(dataset=self.ds, 
                                                                              lengths=[n_train_items, n_test_items, len(self.ds) - n_train_items - n_test_items],
                                                                              generator=g_cpu)
            
            case 'STL10':
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_train_images']), 
                                        image_chw=image_chw,
                                        data_sample_interval=data_train_sample_interval,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id,
                                        path_labels=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_labels']),
                                        path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_text_embeds']),
                                        dataset_name=dataset_name,
                                        n_clients=n_clients)
                self.ds_test = []
                
            case _:
                raise ValueError(f'Unknown dataset: {dataset_name}')
        
        # Log the current device name if CUDA is available
        if torch.cuda.is_available():
            LOGGER.info(f"Current device name of {self.id}: {device} | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else 'cpu')
    
    @property
    def path_save_model(self):
        path = self.model.path_save_model.replace('.pt',f"_tc-{self.t_cut_ratio}.pt")
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

        if len(self.ds_test) > 1:
            # Create the data loader for testing
            self.dl_test = DataLoader(
                self.ds_test,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

    def compute_scores(self, results_folder: str, dir_name: str) -> tuple[float, float, float]:
        
        LOGGER.info(f'Computing scores for t_cut-{int(self.t_cut)}-{self.id}: {dir_name}...')
        fid_score = fid.compute_fid(fdir1=os.path.join(results_folder, f'testing/{int(self.t_cut)}/{dir_name}/{self.id}/'), 
                                        fdir2=os.path.join(results_folder, f'testing/{int(self.t_cut)}/real/{self.id}/'), 
                                        mode="clean", 
                                        model_name="inception_v3")
        
        clip_fid_score = fid.compute_fid(fdir1=os.path.join(results_folder, f'testing/{int(self.t_cut)}/{dir_name}/{self.id}/'), 
                                    fdir2=os.path.join(results_folder, f'testing/{int(self.t_cut)}/real/{self.id}/'), 
                                    mode="clean", 
                                    model_name="clip_vit_b_32")
        
        kid_score = fid.compute_kid(fdir1=os.path.join(results_folder, f'testing/{int(self.t_cut)}/{dir_name}/{self.id}/'), 
                                    fdir2=os.path.join(results_folder, f'testing/{int(self.t_cut)}/real/{self.id}/'), 
                                    mode="clean")
        
        LOGGER.info(f'CLEAN_FID_SCORE-{self.id}-{dir_name}: {fid_score}')
        LOGGER.info(f'CLEAN_CLIP_FID_SCORE-{self.id}-{dir_name}: {clip_fid_score}')
        LOGGER.info(f'CLEAN_KID_SCORE-{self.id}-{dir_name}: {kid_score}')
        
        return fid_score, clip_fid_score, kid_score