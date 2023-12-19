import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import wandb
import numpy as np
import random

from torch.optim import Adam
import torch

from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from collections import namedtuple
from PIL import Image

from denoising_diffusion_pytorch.version import __version__

from src.components.utils.settings import Settings
from src.components.utils import functions as func
from src.components.model.unet import Unet

import logging
logging.getLogger('apscheduler.executors.default').propagate = False

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x0'])

SETTINGS = Settings()
LOGGER=SETTINGS.logger()

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

wandb.login()

# Diffusion trainer class
class Diffusion_Trainer(object):
    def __init__(self,
                 clients: dict,
                 cloud,
                 n_epochs: int,
                 batch_size: int,
                 num_workers: int,
                 loss_lambda: float,
                 initial_loss: float,
                 *,
                 display=False,
                 lr=1e-4,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=25,
                 results_folder='./results',
                 calculate_performance=True,
                 offset_noise_strength=None,
                 ):

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0
        self.initial_loss=initial_loss

        # FID-score computation
        self.calculate_performance = calculate_performance

        # sampling and training hyperparameters
        assert func.has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # Parameters
        self.adam_betas = adam_betas
        self.results_folder = results_folder
        self.offset_noise_strength = offset_noise_strength

        # Nodes
        self.clients = clients
        self.cloud = cloud

        # Model
        self.best_loss = float("inf")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.loss_lambda = loss_lambda

        # Visualization
        self.display = display
        
        # build trainer
        self.build()

    def build(self):
        # Initialization of nodes
        self.cloud.optimizer = Adam(self.cloud.model.parameters(), lr=self.lr, betas=self.adam_betas)
        self.cloud.energy_resources=0
        LOGGER.info(f'{self.cloud.id} Device: {self.cloud.device}')
        
        self.max_ds_length = 0

        for client_id, client in self.clients.items():
            client.set_dl(batch_size=self.batch_size, num_workers=self.num_workers)
            client.t_cut = int(np.round(client.model.num_timesteps * client.t_cut_ratio))
            
            client.optimizer = Adam(client.model.parameters(), self.lr, betas=self.adam_betas)
            client.loss = self.initial_loss
            client.energy_resources = 0
            
            if client.model_type == 'IMAGEN':
                assert not (len(client.model.unets) > 1 and not func.exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
                unet_number = func.default(client.model.only_train_unet_number, 1)
                unet_index = unet_number - 1 
                client.noise_scheduler = client.model.noise_schedulers[unet_index]  
                self.cloud.noise_scheduler = self.cloud.model.noise_schedulers[unet_index]
                   
            if len(client.ds_train) > self.max_ds_length:
                self.max_ds_length = len(client.ds_train)
                
            LOGGER.info(f'{client_id} Device: {client.device}')
        
    def save(self, path_save_model, model, optimizer):

        data = {
            'step': self.step,
            'model': model.state_dict(),
            'opt': optimizer.state_dict()
        }

        torch.save(data, path_save_model)

    def load(self):        
        
        #* CLOUD
        data_cloud = torch.load(self.cloud.model.path_save_model, 
                                map_location=self.cloud.device)
        self.cloud.model.load_state_dict(data_cloud['model'])
        self.cloud.optimizer.load_state_dict(data_cloud['opt'])
        
        #* CLIENTS
        for client_id, client in self.clients.items():
            data_client = torch.load(client.path_save_model, 
                                     map_location=client.device)
            client.model.load_state_dict(data_client['model'])
            client.optimizer.load_state_dict(data_client['opt'])
        
        self.step = data_cloud['step']
            
    def train(self):

        # 1. Start a W&B Run
        self.run = wandb.init(
            project="distributed_genai",
            notes="This experiment will train distributed diffusion models to investigate the effectiveness regarding information inclosure, performance and resources",
            tags=["ecis", "train", "experiment"],
            name=f'train_{datetime.now().strftime("%I:%M%p_%m-%d-%Y")}'
        )
        
        #  Capture a dictionary of the hyperparameters
        wandb.config['TRAINER'] = SETTINGS.diffusion_trainer
        wandb.config['DIFFUSION_MODEL'] = SETTINGS.diffusion_model
        wandb.config['UNET'] = SETTINGS.unet
        wandb.config['CLIENTS'] = SETTINGS.clients

        for epoch in tqdm(range(self.n_epochs), desc=f"Training progress", colour="#00ff00", disable=False):

            epoch_loss = 0.0
            random.shuffle(list(self.clients.keys()))

            for train_step in tqdm(range(int(self.max_ds_length/self.batch_size)), leave=False, desc=f"Epoch {epoch + 1}/{self.n_epochs}", colour="#005500"):
                
                for client_id, client in self.clients.items():
                    total_loss = 0
                    
                    # Loading data
                    img_batch, label_batch = next(iter(client.dl_train))
                    x0 = img_batch.to(client.device)
                    batch_size = len(x0)
                    
                    # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                    # randn_like() returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
                    client.tracker.start_task("DIFFUSION_PROCESS")
                    client.eta = torch.randn_like(x0).to(client.device)
                    
                    # Computing the noisy image based on x0 and the time-step (forward process)
                    match client.model_type:
                        case 'DDPM':
                            t = torch.randint(0, client.model.num_timesteps,
                                        (batch_size,), device=client.device).long()
                            client.noisy_imgs = client.model(x0=x0,
                                                             t=t,
                                                             noise=client.eta,
                                                             offset_noise_strength=self.offset_noise_strength)
                            self.cloud.t = t[t >= client.t_cut].to(self.cloud.device)
                            client.t = t[t < client.t_cut]
                    
                        case 'IMAGEN':
                            t = client.noise_scheduler.sample_random_times(batch_size, device = client.device)
                            client.noisy_imgs, client.log_snr, client.alpha, client.sigma = client.model(x_start=x0,
                                                                                                         times=t,
                                                                                                         noise=client.eta,
                                                                                                         noise_scheduler=client.noise_scheduler,
                                                                                                         random_crop_size=None)
                            self.cloud.t = t[t >= client.t_cut_ratio].to(self.cloud.device)
                            client.t = t[t < client.t_cut_ratio]

                    client.energy_usage['DIFFUSION_PROCESS'] = client.tracker.stop_task(task_name='DIFFUSION_PROCESS')
                    
                    # Getting model estimation of noise based on the images and the time-step
                    self.cloud.t_reshape = self.cloud.t.reshape(-1).long()
                    client.t_reshape = client.t.reshape(-1).long()                   
                    
                    LOGGER.debug(f'client x0 shape: {x0.shape}')
                    LOGGER.debug(f'client t_reshape shape: {client.t_reshape.shape}')
                    LOGGER.debug(f'cloud t_reshape shape: {self.cloud.t_reshape.shape}')

                    #* Cloud Denoising (from t_cut to T)
                    self.cloud.tracker.start_task('DENOISING_PROCESS')
                    if len(self.cloud.t) > 0:
                        match self.cloud.model_type:
                            case 'DDPM':
                                self.cloud.loss = self.cloud.model.backward(noisy_imgs=client.noisy_imgs[client.t.shape[0]:].to(self.cloud.device),
                                                                            imgs=x0.to(self.cloud.device),
                                                                            t=self.cloud.t_reshape,
                                                                            noise=client.eta[client.t.shape[0]:].to(self.cloud.device))
                            case 'IMAGEN':
                                self.cloud.loss = self.cloud.model.backward(x_noisy=client.noisy_imgs[client.t.shape[0]:],
                                                                            images=x0[client.t.shape[0]:],
                                                                            times=self.cloud.t,
                                                                            noise=client.eta[client.t.shape[0]:],
                                                                            log_snr=client.log_snr[client.t.shape[0]:],
                                                                            alpha=client.alpha,
                                                                            sigma=client.sigma,
                                                                            noise_scheduler=self.cloud.noise_scheduler, 
                                                                            text_embeds=label_batch[0][client.t.shape[0]:],
                                                                            text_masks=label_batch[1][client.t.shape[0]:])
                    else:
                        self.cloud.loss = None
                    self.cloud.energy_usage['DENOISING_PROCESS'] = self.cloud.tracker.stop_task(task_name='DENOISING_PROCESS')
                    
                    #* Client Denoising (from zero to t_cut)
                    client.tracker.start_task('DENOISING_PROCESS')
                    if len(client.t) > 0:
                        match self.cloud.model_type:
                            case 'DDPM':
                                client.loss = client.model.backward(noisy_imgs=client.noisy_imgs[:client.t.shape[0]],
                                                                                imgs=x0,
                                                                                t=client.t_reshape,
                                                                                noise=client.eta[:client.t.shape[0]])
                            case 'IMAGEN':
                                client.loss = client.model.backward(x_noisy=client.noisy_imgs[:client.t.shape[0]].to(client.device),
                                                                    images=x0[:client.t.shape[0]].to(client.device),
                                                                    times=client.t_reshape,
                                                                    noise=client.eta[:client.t.shape[0]].to(client.device),
                                                                    log_snr=client.log_snr[:client.t.shape[0]].to(client.device),
                                                                    alpha=client.alpha,
                                                                    sigma=client.sigma,
                                                                    noise_scheduler=client.noise_scheduler, 
                                                                    text_embeds=label_batch[0][:client.t.shape[0]].to(client.device),
                                                                    text_masks=label_batch[1][:client.t.shape[0]].to(client.device))
                    else:
                        client.loss = None
                    client.energy_usage['DENOISING_PROCESS'] = client.tracker.stop_task(task_name='DENOISING_PROCESS')

                    #* Cloud Update
                    self.cloud.tracker.start_task('DDPM_UPDATE')
                    if self.cloud.loss:
                        self.cloud.loss.backward()
                        self.cloud.optimizer.step()
                        self.cloud.optimizer.zero_grad()
                    self.cloud.energy_usage['DDPM_UPDATE'] = self.cloud.tracker.stop_task(task_name='DDPM_UPDATE')
                    
                    #* Client Update                 
                    client.tracker.start_task('DDPM_UPDATE')
                    if client.loss:
                        client.loss.backward()
                        client.optimizer.step()
                        client.optimizer.zero_grad()
                    client.energy_usage['DDPM_UPDATE'] = client.tracker.stop_task(task_name='DDPM_UPDATE')
                    
                    # Aggregate the loss of client and cloud model: lambda * client_loss + (1-lambda) * cloud_loss
                    if self.cloud.loss and client.loss:
                        loss = (self.loss_lambda * client.loss + (1-self.loss_lambda) * self.cloud.loss.to(client.device))
                    elif self.cloud.loss:
                        loss = self.cloud.loss
                    elif client.loss:
                        loss = client.loss
                    total_loss += loss.item()
                    self.step += 1

                # log metrics to wandb
                client_losses = {f'{c_id} loss':c_node.loss for c_id, c_node in self.clients.items()}
                client_energy_diffusion_process = {f'{c_id} diffusion energy':c_node.energy_usage['DIFFUSION_PROCESS'].energy_consumed for c_id, c_node in self.clients.items()}
                client_energy_denoising_process = {f'{c_id} denoising energy':c_node.energy_usage['DENOISING_PROCESS'].energy_consumed for c_id, c_node in self.clients.items()}
                client_energy_ddpm_update = {f'{c_id} ddpm update':c_node.energy_usage['DDPM_UPDATE'].energy_consumed for c_id, c_node in self.clients.items()}
                client_gpu_power_diffusion_process = {f'{c_id} diffusion energy':c_node.energy_usage['DIFFUSION_PROCESS'].gpu_power for c_id, c_node in self.clients.items()}
                client_gpu_power_denoising_process = {f'{c_id} denoising energy':c_node.energy_usage['DENOISING_PROCESS'].gpu_power for c_id, c_node in self.clients.items()}
                client_gpu_power_ddpm_update = {f'{c_id} ddpm update':c_node.energy_usage['DDPM_UPDATE'].gpu_power for c_id, c_node in self.clients.items()}
                wandb.log({"cloud loss": self.cloud.loss, 
                            "total loss": total_loss, 
                            "cloud denoising energy": self.cloud.energy_usage['DENOISING_PROCESS'].energy_consumed,
                            "cloud ddpm update energy": self.cloud.energy_usage['DDPM_UPDATE'].energy_consumed,
                            "cloud denoising gpu power": self.cloud.energy_usage['DENOISING_PROCESS'].gpu_power,
                            "cloud ddpm update gpu power": self.cloud.energy_usage['DDPM_UPDATE'].gpu_power,
                            **client_losses, 
                            **client_energy_diffusion_process,
                            **client_energy_denoising_process,
                            **client_energy_ddpm_update,
                            **client_gpu_power_diffusion_process,
                            **client_gpu_power_denoising_process,
                            **client_gpu_power_ddpm_update,
                            })

            epoch_loss = total_loss

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

            # Storing the model
            if self.best_loss > epoch_loss:
                self.best_loss = epoch_loss
                
                # Display images generated at this epoch
                if self.display:
                    self.generate_images()
                
                for client_id, client in self.clients.items():
                    self.save(model=client.model, 
                              optimizer=client.optimizer,
                              path_save_model=client.path_save_model)
                
                    wandb.log_artifact(artifact_or_path=client.path_save_model,
                                        name=client_id)
                    
                self.save(model=self.cloud.model,
                           optimizer=self.cloud.optimizer,
                           path_save_model=self.cloud.model.path_save_model)
                
                wandb.log_artifact(artifact_or_path=self.cloud.model.path_save_model,
                                   name=self.cloud.id)
                log_string += " --> Best model ever (stored)"
                LOGGER.info(log_string)

        for client_id, client in self.clients.items():
            self.save(model=client.model, 
                        optimizer=client.optimizer,
                        path_save_model=client.path_save_model)
        
            wandb.log_artifact(artifact_or_path=client.path_save_model,
                                name=client_id)
            
        self.save(model=self.cloud.model,
                    optimizer=self.cloud.optimizer,
                    path_save_model=self.cloud.model.path_save_model)
        
        wandb.log_artifact(artifact_or_path=self.cloud.model.path_save_model,
                            name=self.cloud.id)
        log_string += " --> Best model ever (stored)"
        
        wandb.finish()

    def test(self):
        
        # 1. Start a W&B Run
        self.run = wandb.init(
            project="distributed_genai",
            notes="This experiment will trest distributed diffusion models to investigate the effectiveness regarding information inclosure, performance and resources",
            tags=["ecis", "experiment", "test", f"{int(self.clients['CLIENT_1'].t_cut_ratio * 100)}"],
            name=f'test_{datetime.now().strftime("%I:%M%p_%m-%d-%Y")}'
        )
        
        #  Capture a dictionary of the hyperparameters
        wandb.config['TRAINER'] = SETTINGS.diffusion_trainer
        wandb.config['DIFFUSION_MODEL'] = SETTINGS.diffusion_model
        wandb.config['UNET'] = SETTINGS.unet
        wandb.config['CLIENTS'] = SETTINGS.clients
        
        self.load()
        #self.generate_images(testing=True)
        
        for client_id, client in self.clients.items():
            client.compute_performance()
            client.compute_information_disclosure()
        
        wandb.log({f'Performance {client_id}_train | KID' : client.kid_score_train  for client_id, client in self.clients.items()})
        wandb.log({f'Performance {client_id}_test | KID' : client.kid_score_test  for client_id, client in self.clients.items()})
        wandb.log({f'Information Disclosure {client_id}_train | KID' : client.kid_inf_dis_train  for client_id, client in self.clients.items()})
        wandb.log({f'Information Disclosure {client_id}_test | KID' : client.kid_inf_dis_test  for client_id, client in self.clients.items()})
        wandb.log({f'Information Disclosure {client_id}_train | Pixel_Comparison_MSE (AGGREGATED)' : client.inf_dis_mse  for client_id, client in self.clients.items()})
        wandb.log({f'Information Disclosure {client_id}_train | Pixel_Comparison_MSE (MEAN)' : client.inf_dis_mse_mean  for client_id, client in self.clients.items()})
    
        wandb.finish()
    
    def generate_images(self, testing=False):
        
        # Obtain SETTINGS
        sample_batch_size=SETTINGS.diffusion_trainer['GENERATION']['sample_batch_size']
        return_all_timesteps=SETTINGS.diffusion_trainer['GENERATION']['return_all_timesteps']
        n_samples=SETTINGS.diffusion_trainer['GENERATION']['n_samples']
        text_embeds=SETTINGS.diffusion_trainer['GENERATION']['text_embeds'] # TODO
        
        for batch_k in range(int(np.ceil(n_samples/sample_batch_size))):
            
            if not SETTINGS.diffusion_trainer['GENERATION']['create_new_samples'] and testing:
                continue
            
            # Create Noise Image
            shape = (sample_batch_size, self.cloud.model.channels, self.cloud.model.image_height, self.cloud.model.image_width)
            noise_img = torch.randn(shape, device=self.cloud.device)
            
            # Sample using Cloud
            match self.cloud.model_type:
                case 'DDPM':
                    cloud_img_samples = self.cloud.model.sample(batch_size=sample_batch_size,
                                                                t_min=0,
                                                                t_max=self.cloud.model.num_timesteps,
                                                                noise_img=noise_img,
                                                                return_all_timesteps=True)
                case 'IMAGEN':
                    cloud_img_samples = self.cloud.model.sample(t_min=0,
                                                                t_max=1,
                                                                noise_img=noise_img,
                                                                return_all_timesteps=True,
                                                                text_embeds=text_embeds)
            
            # Continue sampling using Clients
            for client_id, client in self.clients.items():
                
                if client.t_cut_ratio == 0:
                    client_img_samples = cloud_img_samples[:,-1,...]
                    
                elif client.t_cut_ratio == 1:
                    match self.cloud.model_type:
                        case 'DDPM':
                            client_img_samples = client.model.sample(batch_size=sample_batch_size,
                                                        t_min=0,
                                                        t_max=client.model.num_timesteps,
                                                        noise_img=noise_img.to(client.device),
                                                        return_all_timesteps=return_all_timesteps)
                        case 'IMAGEN':
                            client_img_samples = client.model.sample(t_min=0,
                                                        t_max=1,
                                                        noise_img=noise_img.to(client.device),
                                                        return_all_timesteps=return_all_timesteps,
                                                        text_embeds=text_embeds)
                    # Store images
                    if return_all_timesteps:
                        img_samples=torch.cat([cloud_img_samples[:,:-(client.t_cut+1),...].to(client.device), client_img_samples], dim=1)
                    else:
                        img_samples=client_img_samples
                            
                else:
                    match self.cloud.model_type:
                        case 'DDPM':
                            client_img_samples = client.model.sample(batch_size=sample_batch_size,
                                                        t_min=0,
                                                        t_max=client.t_cut,
                                                        noise_img=cloud_img_samples[:,-(client.t_cut+1),...].to(client.device),
                                                        return_all_timesteps=return_all_timesteps)
                        case 'IMAGEN':
                            client_img_samples = client.model.sample(t_min=0,
                                                        t_max=client.t_cut_ratio,
                                                        noise_img=cloud_img_samples[:,-(client.t_cut+1),...].to(client.device),
                                                        return_all_timesteps=return_all_timesteps,
                                                        text_embeds=text_embeds)
                    # Store images
                    if return_all_timesteps:
                        img_samples=torch.cat([cloud_img_samples[:,:-(client.t_cut+1),...].to(client.device), client_img_samples], dim=1)
                    else:
                        img_samples=client_img_samples
                    
                #* Client images
                for batch_idx, imgs in enumerate(img_samples):
                    wandb_table = wandb.Table(
                        columns=['Resource', 'Generated-Images']
                    )
                    image_idx=batch_k*sample_batch_size + batch_idx
                    imgs_raw=(func.unnormalize_to_zero_to_one(imgs.cpu().numpy().squeeze()) * 255).astype(np.uint8)
                    
                    if testing:
                        image_save_path=self.results_folder + f"testing/{int(client.t_cut_ratio*100)}/image_{client_id}_{image_idx}.png"
                    else:
                        image_save_path=self.results_folder + f"training/image_{client_id}_{image_idx}.png"
                    
                    if return_all_timesteps:
                        for timestep_idx, img in enumerate(imgs_raw):
                            wandb_table.add_data('CLOUD' if timestep_idx < self.cloud.model.num_timesteps-client.t_cut else client_id, wandb.Image(img))
                            
                            image_save_path=image_save_path.remove('.png',f'-{timestep_idx}.png')
                            img = Image.fromarray(img)
                            img.save(image_save_path)
                        
                        wandb.log({f'Generated-Images-Table_{client_id}_{batch_idx}': wandb_table})
                    
                    else:
                        img = Image.fromarray(imgs_raw)
                        img.save(image_save_path)
                
                #* Cloud Images
                for batch_idx, cloud_imgs in enumerate(cloud_img_samples[:,-(client.t_cut+1),...]):
                    wandb_table = wandb.Table(
                        columns=['Resource', 'Generated-Images']
                    )
                    image_idx=batch_k*sample_batch_size + batch_idx
                    imgs_raw=(func.unnormalize_to_zero_to_one(cloud_imgs.cpu().numpy().squeeze()) * 255).astype(np.uint8)
                    
                    if testing:
                        image_save_path=self.results_folder + f"testing/{int(client.t_cut_ratio*100)}_cloud/image_{client_id}_{image_idx}.png"
                    else:
                        image_save_path=self.results_folder + f"training/image_{client_id}_{image_idx}_cloud.png"
                    
                    img = Image.fromarray(imgs_raw)
                    img.save(image_save_path)
