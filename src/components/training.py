import sys
import os
sys.path.append(os.path.abspath(os.curdir))

# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST, FashionMNIST

from src.components.utils.settings import Settings
from src.components.visualization.display_images import show_first_batch, show_images
from src.components.model.ddpm import DDPM_Trainer
from src.components.model.unet import UNet
from src.components.nodes.collaborator_node import Collaborator
from src.components.nodes.cloud_node import Cloud


SETTINGS = Settings()

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"
no_train = False
fashion = False
batch_size = 128
n_epochs = 20
lr = 0.001
store_path = "./src/assets/ddpm_fashion.pt" if fashion else "./src/assets/ddpm_mnist.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS.logger.info(f'Using device: {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'cpu'))

collaborators=dict()
for dataset in [MNIST, FashionMNIST]:
    collaborators['MNIST'] = Collaborator(device=device, dataset_function=MNIST)
    collaborators['FashionMNIST'] = Collaborator(device=device, dataset_function=FashionMNIST)
cloud=Cloud(device=device)

# Optionally, show a batch of regular images
#show_first_batch(collaborator.dl)

# Defining model

# Optionally, show the diffusion (forward) process
# show_forward(ddpm, loader, device)

# Optionally, show the denoising (backward) process 
# generated = ddpm.generate_new_images(gif_name="before_training.gif")
# show_images(generated, "Images generated before training")

ddpm_trainer = DDPM_Trainer(collaborators=collaborators, cloud=cloud, **SETTINGS.ddpm_trainer['DEFAULT'])
ddpm_trainer.train()
