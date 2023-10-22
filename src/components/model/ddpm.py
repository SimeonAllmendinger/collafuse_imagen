import sys
import os
sys.path.append(os.path.abspath(os.curdir))

# DDPM class
import einops
import random
import torch
import torch.nn as nn

import numpy as np

from tqdm.auto import tqdm
from torch.optim import Adam

from src.components.utils.settings import Settings
from src.components.model.unet import UNet

from src.components.visualization.display_images import show_images

SETTINGS=Settings()

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class DDPM(nn.Module):
    def __init__(self, network, device: str, n_steps: int, min_beta: float, max_beta: float, image_chw: set, path_save_model="ddpm_model.pt"):
        super(DDPM, self).__init__()
        
        self.device = device
        self.path_save_model=path_save_model
        
        self.n_steps = n_steps
        self.c, self.h, self.w = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t: int, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy_imgs = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy_imgs

    def backward(self, x: torch.Tensor, t: torch.Tensor):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x=x, t=t)
    
    def generate_new_images(self, n_samples=16, frames_per_gif=100, gif_name="sampling.gif"):
        frame_idxs = np.linspace(0, self.n_steps, frames_per_gif).astype(np.uint)
        frames = []

        with torch.no_grad():

            # Starting from random noise
            x = torch.randn(n_samples, self.c, self.h, self.w).to(self.device)

            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device).long()
                eta_theta = self.backward(x=x, t=time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, self.c, self.h, self.w).to(device)

                    # Option 1: sigma_t squared = beta_t
                    if True:
                        beta_t = self.betas[t]
                        sigma_t = beta_t.sqrt()
                    
                    # Option 2: sigma_t squared = beta_tilda_t
                    elif False:
                        prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                        sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

                # Adding frames to the GIF
                if idx in frame_idxs or t == 0:
                    # Putting digits in range [0, 255]
                    normalized = x.clone()
                    for i in range(len(normalized)):
                        normalized[i] -= torch.min(normalized[i])
                        normalized[i] *= 255 / torch.max(normalized[i])

                    # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)

                    # Rendering frame
                    frames.append(frame)

        '''# Storing the gif
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])'''
        return x


class DDPM_Trainer():
    def __init__(self, collaborators: dict, cloud, n_epochs: int, lr: float, batch_size: int, t_cut_ratio: int, loss_alpha: float, display=False):
        # Nodes
        self.collaborators = collaborators
        self.cloud = cloud
        
        # Model
        self.mse = nn.MSELoss()
        self.best_loss = float("inf")
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.lr = lr
        self.t_cut=int(np.round(self.cloud.ddpm.n_steps*t_cut_ratio))
        self.loss_alpha = loss_alpha
        
        # Visualization
        self.display=display
        

        
    def train(self):
        
        self.cloud.optimizer = Adam(self.cloud.ddpm.parameters(), self.lr)
        
        for epoch in tqdm(range(self.n_epochs), desc=f"Training progress", colour="#00ff00"):
                
            epoch_loss = 0.0
            random.shuffle(list(self.collaborators.keys())) 

            for id, collaborator in self.collaborators.items():
                collaborator.set_dl(batch_size=self.batch_size)
                collaborator.optimizer = Adam(collaborator.ddpm.parameters(), self.lr)
                
                for step, batch in enumerate(tqdm(collaborator.dl, leave=False, desc=f"Epoch {epoch + 1}/{self.n_epochs}", colour="#005500")):
                
                    # Loading data
                    collaborator.x0 = batch[0].to(collaborator.ddpm.device)
                    collaborator.n = len(collaborator.x0)

                    # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                    # randn_like() returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. 
                    collaborator.eta = torch.randn_like(collaborator.x0).to(collaborator.ddpm.device)
                    t = torch.randint(0, collaborator.ddpm.n_steps, (collaborator.n,)).to(collaborator.ddpm.device)

                    # Computing the noisy image based on x0 and the time-step (forward process)
                    collaborator.noisy_imgs = collaborator.ddpm(collaborator.x0, t, collaborator.eta)

                    # Getting model estimation of noise based on the images and the time-step
                    self.cloud.t=t[t<self.t_cut]
                    self.cloud.t_reshape=self.cloud.t.reshape(self.cloud.t.shape[0], -1)
                    collaborator.t=t[t>=self.t_cut]
                    collaborator.t_reshape=collaborator.t.reshape(collaborator.t.shape[0], -1)
                    
                    eta_theta_cloud = self.cloud.ddpm.backward(collaborator.noisy_imgs[:self.cloud.t.shape[0]], self.cloud.t_reshape)
                    eta_theta_colab = collaborator.ddpm.backward(collaborator.noisy_imgs[self.cloud.t.shape[0]:], collaborator.t_reshape)
                    
                    # Optimizing the MSE between the noise plugged and the predicted noise
                    loss = self.loss_alpha * self.mse(eta_theta_cloud, collaborator.eta[:self.cloud.t.shape[0]]) + (1-self.loss_alpha) * self.mse(eta_theta_colab, collaborator.eta[self.cloud.t.shape[0]:])
                    collaborator.optimizer.zero_grad()
                    self.cloud.optimizer.zero_grad()
                    loss.backward()
                    collaborator.optimizer.step()
                    self.cloud.optimizer.step()
                    
                    collaborator.loss = loss.item()

            epoch_loss += loss.item() * len(collaborator.x0) / len(collaborator.ds)

            # Display images generated at this epoch
            if self.display:
                show_images(self.ddpm.generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

            # Storing the model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(self.ddpm.state_dict(), self.ddpm.path_save_model)
                log_string += " --> Best model ever (stored)"

            SETTINGS.logger.info(log_string)
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddpm = DDPM(network=UNet(**SETTINGS.unet['DEFAULT']), device=device, **SETTINGS.ddpm['DEFAULT'])