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
from cleanfid import fid, features, clip_features
from cleanfid import utils as fid_utils

from torch.utils.data import DataLoader, Dataset
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
        data_sample_interval: list, # interval of data sample
        t_cut_ratio: float,
        client_id: str,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        is_dataset_results=False,
        is_dataset_cloud=False,
        path_labels=None,
        path_text_embeds=None,
        dataset_name=None,
        celeb_a_attributes=None,
        birds_attributes=None
        ):
        
        super().__init__()
        self.folder = folder
        self.dataset_name=dataset_name
        self.client_id = client_id
        self.path_text_embeds = path_text_embeds
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
        '''if is_dataset_results:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'{int(t_cut_ratio*100)}/image_{client_id}_*.{ext}')]
        elif is_dataset_cloud:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'{int(t_cut_ratio*100)}_cloud/image_{client_id}_*.{ext}')]
        else:'''
        
        match self.dataset_name:
            case 'CelebA':
                    
                df = pd.read_csv(path_labels, sep=r'\s+', header=1, usecols=['PATH'] + celeb_a_attributes)
                
                # Filter rows where at least one of the specified columns has a value of 1
                df = df[df.eq(1).any(axis=1)]
                self.paths = df['PATH'].to_list()
                
                # Extract column names with at least one '1' and convert the list of column names to a string
                self.texts = df.apply(lambda row: ', '.join(row.index[row == 1].tolist()).replace('_', ' '), axis=1).to_list()
                
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
                df_labels = pd.read_csv('/home/woody/btr0/btr0104h/data/Birds/attributes/image_attribute_labels.txt', sep=r'\s+', header=0, usecols=[0,1,2])
                df_att = pd.read_csv('/home/woody/btr0/btr0104h/data/Birds/attributes/attributes.txt', sep=' ', header=0)
                
                df_labels = df_labels.loc[(df_labels.iloc[:,1].isin(birds_attributes)) & (df_labels.iloc[:,2] == 1),:]
                list_img_unique = sorted(df_labels.iloc[:,0].unique().tolist())
                
                self.paths = df_img.loc[df_img.iloc[:,0].isin(list_img_unique),:].iloc[:,1].values.tolist()
                self.texts = [', '.join([df_att.iloc[class_idx,1].replace('_',' ').replace('::', ' ') for class_idx in sorted(df_labels.loc[(df_labels.iloc[:,0] == img_idx),:].iloc[:,1].values.tolist())]) for img_idx in sorted(list_img_unique)]
                
                for path in self.paths:
                    img = Image.open(os.path.join(self.folder,path))
                    img= self.transform(img)
                    if img.shape[0] == 1:
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
                        max_mask_size = 25
                        
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
                self.paths = [p for ext in exts for k in range(self.data_sample_min, self.data_sample_max) for p in Path(f'{folder}').glob(f'**/{k:03d}-*.{ext}')]
        
        '''# define inception frame size
        self.inception_width = 299
        self.inception_height = 299
        
        # compute center offset
        self.inception_x_center = (self.inception_width - self.image_chw[2]) // 2
        self.inception_y_center = (self.inception_height - self.image_chw[1]) // 2'''
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        text = self.texts[index]
        
        match self.dataset_name:
            case 'STL10':
                label = (self.text_embeds[index], self.text_masks[index])
            case 'CelebA':
                label = (self.text_embeds[index], self.text_masks[index])
            case 'Birds':
                label = (self.text_embeds[index], self.text_masks[index])
          
        img = Image.open(path)
        img= self.transform(img)
        
        assert img.shape[0] != 1, f'{path} | {img.shape[0]}'
        
        '''# create new image of desired size and color (blue) for padding
        inception_frame = np.zeros((self.inception_height, self.inception_width), dtype=np.uint8)
        
        # copy img image into center of result image
        inception_frame[self.inception_y_center:self.inception_y_center+self.image_chw[1], self.inception_x_center:self.inception_x_center+self.image_chw[2]] = func.unnormalize_to_zero_to_one(img.cpu().numpy().squeeze()* 255).astype(np.uint8)
        inception_frame=np.stack((inception_frame,)*3, axis=-1)
        inception_img=torch.tensor(func.normalize_to_neg_one_to_one(inception_frame)).permute(2,0,1)'''
        
        return img, label, text #inception_img
    
class Client(BaseNode):
    def __init__(self, 
                idx: int,
                device, 
                dataset_name: str,
                image_chw: int,
                data_train_sample_interval: list,
                data_test_sample_interval: list,
                t_cut_ratio: float,
                path_tmp_dir: str,
                model_type: str,
                celeb_a_attributes: [str] = None,
                birds_attributes: [str] = None
                ):
        
        # Call the parent class constructor
        super().__init__(id=f'CLIENT_{idx}', node_type='Client', device=device, model_type=model_type, dataset_name=dataset_name)
        
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
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_train_sliced']), 
                                        image_chw=image_chw,
                                        data_sample_interval=data_train_sample_interval,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id)
                self.ds_test = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['BraTS2020']['path_test_sliced']), 
                                    image_chw=image_chw,
                                    data_sample_interval=data_test_sample_interval,
                                    t_cut_ratio=self.t_cut_ratio,
                                    client_id=self.id)
            case 'CelebA':
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_train_images']), 
                                        image_chw=image_chw,
                                        data_sample_interval=data_train_sample_interval,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id,
                                        path_labels=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_labels']),
                                        path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['CelebA']['path_text_embeds']),
                                        dataset_name=dataset_name,
                                        celeb_a_attributes=celeb_a_attributes)
                self.ds_test = []
            
            case 'Birds':
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_train_images']), 
                                        image_chw=image_chw,
                                        data_sample_interval=data_train_sample_interval,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id,
                                        path_labels=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_labels']),
                                        path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['Birds']['path_text_embeds']),
                                        dataset_name=dataset_name,
                                        birds_attributes=birds_attributes)
                self.ds_test = []
            
            case 'STL10':
                self.ds_train = Dataset(folder=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_train_images']), 
                                        image_chw=image_chw,
                                        data_sample_interval=data_train_sample_interval,
                                        t_cut_ratio=self.t_cut_ratio,
                                        client_id=self.id,
                                        path_labels=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_labels']),
                                        path_text_embeds=os.path.join(path_tmp_dir,SETTINGS.data['STL10']['path_text_embeds']),
                                        dataset_name=dataset_name)
                self.ds_test = []
                
            case _:
                raise ValueError(f'Unknown dataset: {dataset_name}')
        
        # Log the length of the train dataset
        LOGGER.debug(f'Train Dataset length: {len(self.ds_train)}')
        
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
        kid_feature_model = features.build_feature_extractor(mode="clean", device=torch.device("cuda:0"))
        kid_features_actual, kid_features_synthetic, kid_features_test_data = self.get_inf_dis_features(feature_model=kid_feature_model)

        self.kid_inf_dis_train = fid.kernel_distance(feats1=kid_features_actual, feats2=kid_features_synthetic, num_subsets=10, max_subset_size=400)
        self.kid_inf_dis_test = fid.kernel_distance(feats1=kid_features_test_data, feats2=kid_features_synthetic, num_subsets=10, max_subset_size=400)
        
        self.inf_dis_mse = 0
        n_images = 0
        for batch_cloud in tqdm(self.dl_inf_dis):
            for batch_train in  self.dl_train:
                for image_cloud in batch_cloud[0]:
                    for image_train in batch_train[0]:
                        image_mse = torch.sum((image_train - image_cloud) ** 2) / (self.image_chw[0] * self.image_chw[1] * self.image_chw[2])
                        self.inf_dis_mse += image_mse.item()
                        n_images += 1
                        
        self.inf_dis_mse_mean = self.inf_dis_mse / n_images
        
    @torch.inference_mode()
    def get_performance_features(self, feature_model):
        
        # Data
        if not self.dl_test:
            self.set_dl(batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'], 
                        num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
            
        ds_samples = Dataset(folder=os.path.join(SETTINGS.diffusion_trainer['DEFAULT']['results_folder'],'testing'),
                             image_chw=self.image_chw,
                             data_sample_interval=None,
                             is_dataset_results=True,
                             t_cut_ratio=self.t_cut_ratio,
                             client_id=self.id)
        
        dl_samples = DataLoader(ds_samples,
                                batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'],
                                shuffle=True,
                                num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
        
        feature_list_actual = []
        feature_list_synthetic = []
        feature_list_test_data = []
        
        for batch in self.dl_train:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_actual.append(features)
            
        for batch in self.dl_test:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_test_data.append(features)
        
        for batch in dl_samples:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
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
                             image_chw=self.image_chw,
                             data_sample_interval=None,
                             is_dataset_cloud=True,
                             t_cut_ratio=self.t_cut_ratio,
                             client_id=self.id)
        
        self.dl_inf_dis = DataLoader(ds_inf_dis,
                                batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size'],
                                shuffle=True,
                                num_workers=SETTINGS.diffusion_trainer['DEFAULT']['num_workers'])
        
        feature_list_actual = []
        feature_list_inf_dis = []
        feature_list_test_data = []
        
        for batch in self.dl_train:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_actual.append(features)
            
        for batch in self.dl_test:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_test_data.append(features)
        
        for batch in self.dl_inf_dis:
            features = feature_model(batch[1].to(torch.device("cuda:0"))).detach().cpu().numpy()
            feature_list_inf_dis.append(features)
        
        features_actual = np.concatenate(feature_list_actual)
        features_inf_dis = np.concatenate(feature_list_inf_dis)
        features_test_data = np.concatenate(feature_list_test_data)
        
        return features_actual, features_inf_dis, features_test_data      
    

'''labels = pd.read_csv('data/CelebA/annotations/identity_CelebA.txt', sep=' ', dtype=str).iloc[:,1].to_list()
print(len(labels))
for i in tqdm(range(0, int(np.ceil(len(labels)/40000)))):
    if (i+1)*10000 > len(labels):
        text_embeds = t5_encode_text(labels[i*40000:], device=torch.device(1))
    else:
        text_embeds = t5_encode_text(labels[i*40000:(i+1)*40000], device=torch.device(1))
    print(text_embeds.shape)
    torch.save(text_embeds, f'data/CelebA/annotations/text_embeds_base_{i+1}.pt')
    torch.cuda.empty_cache()'''