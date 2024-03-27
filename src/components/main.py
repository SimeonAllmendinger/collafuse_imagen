import sys
import os
sys.path.append(os.path.abspath(os.curdir))

# Import of libraries
import numpy as np
import argparse
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torchsummary import summary
from torchvision.datasets.mnist import MNIST, FashionMNIST

from src.components.utils.settings import Settings
from src.components.model.diffusion import Diffusion_Trainer
from src.components.nodes.client_node import Client
from src.components.nodes.cloud_node import Cloud
from src.components.utils.data_slicer import slice

parser = argparse.ArgumentParser(
                prog='Collaborative Diffusion Models')

parser.add_argument('--path_tmp_dir',
                    default='./',
                    help='PATH to data directory')

SETTINGS = Settings()
LOGGER=SETTINGS.logger()

def main(path_tmp_dir: str):
    
    LOGGER.info(f'Available GPU devices: {torch.cuda.device_count()}')
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    LOGGER.info(available_gpus)

    clients = {}
    n_clients = len(SETTINGS.clients)
    image_chw = (1, SETTINGS.imagen_model['DEFAULT']['image_sizes'][0], SETTINGS.imagen_model['DEFAULT']['image_sizes'][0])
    #client_device_idx = 0

    LOGGER.info('Start initializing clients...')
    
    for i, (client_name, client_dict) in enumerate(SETTINGS.clients.items()):
        
        # define data intervals
        n_train_items = SETTINGS.data[client_dict['dataset_name']]['n_train_items']
        n_test_items = SETTINGS.data[client_dict['dataset_name']]['n_test_items']
        model_type = client_dict['model_type']
        
        # device
        client_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #client_device_idx = client_device_idx if client_device_idx < len(available_gpus)-1 else 0
        
        client = Client(**client_dict, 
                        device=client_device, 
                        image_chw=image_chw, 
                        n_train_items=n_train_items,
                        n_test_items=n_test_items,
                        path_tmp_dir=path_tmp_dir,
                        n_clients=n_clients)
        
        clients[client.id] = client
        
        LOGGER.info(f'{client_name} created with {len(client.ds_train)} train samples and {len(client.ds_test)} test samples')
    
    cloud_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    cloud = Cloud(**SETTINGS.clouds['CLOUD'], device=cloud_device)
    
    diffusion_trainer = Diffusion_Trainer(clients=clients, cloud=cloud, **SETTINGS.diffusion_trainer['DEFAULT'])
    
    match SETTINGS.diffusion_trainer['mode']:
        case 'train':
            diffusion_trainer.train()
        case 'test':
            diffusion_trainer.test()
    
if __name__ == '__main__':
    args = parser.parse_args()
    path_tmp_dir = args.path_tmp_dir
    LOGGER.info(f'PATH_TMP_DIR: {path_tmp_dir}')
    #slice(dataset_name='BraTS2020', is_training=True)
    #slice(dataset_name='BraTS2020', is_training=False)
    main(path_tmp_dir=path_tmp_dir)