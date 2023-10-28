import sys
import os
sys.path.append(os.path.abspath(os.curdir))

# Import of libraries
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision.datasets.mnist import MNIST, FashionMNIST

from src.components.utils.settings import Settings
from src.components.visualization.display_images import show_first_batch, show_images
from src.components.model.diffusion import Diffusion_Trainer
from src.components.nodes.client_node import Client
from src.components.nodes.cloud_node import Cloud

parser = argparse.ArgumentParser(
                prog='DistributedGenAi',
                epilog='For help refer to uerib@student.kit.edu')

parser.add_argument('--path_data_dir',
                    default='/home/stud01/distributedgenai/',
                    help='PATH to data directory')

SETTINGS = Settings()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS.logger.info(f'Available devices: {torch.cuda.device_count()}')
SETTINGS.logger.info(f'Using general device: {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'cpu'))

clients=dict()
for i, dataset_function in enumerate([MNIST, FashionMNIST]):
    client = Client(idx=(i+1), device=device, dataset_function=dataset_function, image_chw=SETTINGS.diffusion_model['DEFAULT']['image_chw'])
    clients[client.id] = client
cloud=Cloud(device=device)

# Optionally, show a batch of regular images
#show_first_batch(collaborator.dl)

# Defining model

# Optionally, show the diffusion (forward) process
# show_forward(ddpm, loader, device)

# Optionally, show the denoising (backward) process 
# generated = ddpm.generate_new_images(gif_name="before_training.gif")
# show_images(generated, "Images generated before training")

diffusion_trainer = Diffusion_Trainer(clients=clients, cloud=cloud, **SETTINGS.diffusion_trainer['DEFAULT'])
diffusion_trainer.train()
