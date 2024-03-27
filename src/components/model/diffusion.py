import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import wandb
import numpy as np
import pandas as pd
import random

from torch.optim import Adam
import torch

from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from collections import namedtuple
from PIL import Image
from cleanfid import fid

from denoising_diffusion_pytorch.version import __version__

from src.components.utils.settings import Settings
from src.components.utils import functions as func
from src.components.model.unet import Unet
from src.components.evaluation.display_results import visualize_performance

import logging
logging.getLogger('apscheduler.executors.default').propagate = False

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x0'])

SETTINGS = Settings()
LOGGER=SETTINGS.logger()

# Setting reproducibility
SEED = 10
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
                 load_existing_model=False,
                 pretrain_client_from_cloud=False
                 ):

        Path(results_folder).mkdir(exist_ok=True)
        self.results_folder = results_folder

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
        self.load_existing_model=load_existing_model
        self.pretrain_client_from_cloud=pretrain_client_from_cloud

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
                
            if len(client.ds_train) > self.max_ds_length:
                self.max_ds_length = len(client.ds_train)
                
            LOGGER.info(f'{client_id} Device: {client.device}')
        
        if self.cloud.model_type == 'IMAGEN':
            self.cloud.noise_scheduler = self.cloud.model.noise_schedulers[unet_index]
        
        if self.pretrain_client_from_cloud:
            LOGGER.info(f'Pretrain clients from cloud')
            self.load()
        
        elif self.load_existing_model:
            LOGGER.info(f'Load existing models')
            self.load()
            
        else:
            LOGGER.info(f'Initialize new models')
        
    def save(self, path_save_model, model, optimizer):

        data = {
            'step': self.step,
            'model': model.state_dict(),
            'opt': optimizer.state_dict()
        }

        torch.save(data, path_save_model)
        LOGGER.info(f'Model saved at {path_save_model}')

    def load(self):        
        
        if os.path.exists(self.cloud.model.path_save_model):
            #* CLOUD
            data_cloud = torch.load(self.cloud.model.path_save_model, 
                                    map_location=self.cloud.device)
            self.cloud.model.load_state_dict(data_cloud['model'])
            self.cloud.optimizer.load_state_dict(data_cloud['opt'])
            self.step = data_cloud['step']
            LOGGER.info(f'{self.cloud.id} model loaded ({self.cloud.model.path_save_model})')
        else:
            LOGGER.warning(f'Cloud model does not exist at {self.cloud.model.path_save_model}')
        
        #* CLIENTS
        for client_id, client in self.clients.items():
            if os.path.exists(client.path_save_model):
                if self.pretrain_client_from_cloud:
                    #* Pretrain clients from cloud
                    data_client = torch.load(self.cloud.model.path_save_model, 
                                    map_location=self.cloud.device)
                    client.model.load_state_dict(data_client['model'])
                    client.optimizer.load_state_dict(data_client['opt'])
                    LOGGER.info(f'{client_id} model loaded from cloud ({self.cloud.model.path_save_model})')
                else:
                    data_client = torch.load(client.path_save_model, 
                                            map_location=client.device)
                    client.model.load_state_dict(data_client['model'])
                    client.optimizer.load_state_dict(data_client['opt'])
                    LOGGER.info(f'{client_id} model loaded ({client.path_save_model})')
            else:
                LOGGER.warning(f'{client_id} model does not exist at {client.path_save_model}')
            
    def train(self):

        # 1. Start a W&B Run
        self.run = wandb.init(
            project="distributed_genai",
            notes="This experiment will train distributed diffusion models to investigate the effectiveness regarding information inclosure, performance and resources",
            tags=["ECML", "train", "experiment"],
            name=f'train_{datetime.now().strftime("%I:%M%p_%m-%d-%Y")}_{self.cloud.dataset_name}_{self.cloud.id}_clients-{len(self.clients)}_s-{SETTINGS.imagen_model["DEFAULT"]["image_sizes"][0]}'
        )
        
        #  Capture a dictionary of the hyperparameters
        wandb.config['TRAINER'] = SETTINGS.diffusion_trainer
        wandb.config['DIFFUSION_MODEL'] = SETTINGS.diffusion_model
        wandb.config['UNET'] = SETTINGS.unet
        wandb.config['CLIENTS'] = SETTINGS.clients
        wandb.config['IMAGEN'] = SETTINGS.imagen_model
        wandb.config['EFFICIENT_UNET'] = SETTINGS.efficient_unet

        for epoch in tqdm(range(self.n_epochs), desc=f"Training progress", colour="#00ff00", disable=False):

            self.epoch = epoch
            epoch_loss = 0.0
            random.shuffle(list(self.clients.keys()))
            
            for train_step in tqdm(range(10), leave=False, desc=f"Epoch {epoch + 1}/{self.n_epochs}", colour="#005500"):
                total_loss = 0
                
                for client_id, client in self.clients.items():
                    
                    # Loading data
                    img_batch, label_batch, text = next(iter(client.dl_train))
                    x0 = img_batch.cuda()
                    batch_size = len(x0)
                    
                    # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                    # randn_like() returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
                    client_eta = torch.randn_like(x0) # shape: (batch_size,3,64,64)
                    cloud_eta = torch.randn_like(x0) # shape: (batch_size,3,64,64)
                    LOGGER.debug(f'CLIENT ETA: {client_eta.shape}')
                    
                    #! Forward Process
                    # Computing the noisy image based on x0 and the time-step (forward process)
                    match client.model_type:
                        case 'DDPM':
                            if client.t_cut_ratio > 0:
                                # t client
                                client.t = torch.randint(0,client.t_cut,
                                                         (batch_size,), 
                                                         device=client.device).long()
                                # noisy_imgs client
                                client_noisy_imgs = client.model(x0=x0,
                                                                 t=client.t,
                                                                 noise=client_eta.cuda(),
                                                                 offset_noise_strength=self.offset_noise_strength)
                                
                            else:
                                client.t = torch.tensor([]).long()
                                cut_noisy_imgs = x0
                                
                            ## Cloud
                            if client.t_cut_ratio < 1:
                                # t cloud
                                self.cloud.t = torch.randint(client.t_cut,
                                                             self.cloud.model.num_timesteps,
                                                             (batch_size,), 
                                                             device=client.device).long()
                                # t cut
                                t_cut_tensor = torch.full_like(input=self.cloud.t, fill_value=client.t_cut)
                                # cut noisy_imgs
                                cut_noisy_imgs = self.cloud.model(x0=x0,
                                                                  t=t_cut_tensor,
                                                                  noise=client_eta.cuda(),
                                                                  offset_noise_strength=self.offset_noise_strength)
                                
                                # noisy_imgs cloud
                                cloud_noisy_imgs = self.cloud.model(x0=cut_noisy_imgs.cuda(),
                                                                    t=self.cloud.t,
                                                                    noise=cloud_eta.cuda(),
                                                                    offset_noise_strength=self.offset_noise_strength)
                            else:
                                self.cloud.t = torch.tensor([]).long()
                                
                        case 'IMAGEN':
                            ## Client
                            if client.t_cut_ratio > 0:
                                # t client
                                client.t = client.noise_scheduler.sample_random_times(batch_size, sample_interval=(0, client.t_cut_ratio), device = client.device) # shape: (batch_size)
                                # noisy_imgs client
                                client_noisy_imgs, client_log_snr, client_alpha, client_sigma = client.model(x_start=x0,
                                                                                                            times=client.t,
                                                                                                            noise=client_eta.cuda(),
                                                                                                            noise_scheduler=client.noise_scheduler,
                                                                                                            random_crop_size=None) # shape: (batch_size,3,64,64)
                                
                            else:
                                client.t = torch.tensor([]).long()
                                cut_noisy_imgs = x0
                            
                            ## Cloud
                            if client.t_cut_ratio < 1:
                                # t cloud
                                self.cloud.t = self.cloud.noise_scheduler.sample_random_times(batch_size, sample_interval=(client.t_cut_ratio,1),device = client.device) # shape: (batch_size)
                                # t cut
                                t_cut_tensor = torch.full_like(input=self.cloud.t, fill_value=client.t_cut_ratio)
                                # cut noisy_imgs
                                log_snr = self.cloud.noise_scheduler.log_snr(t_cut_tensor).type(x0.dtype)
                                log_snr_padded_dim = func.right_pad_dims_to(x0, log_snr)
                                cut_alpha, cut_sigma =  func.log_snr_to_alpha_sigma(log_snr_padded_dim)
                                
                                cut_noisy_imgs = cut_alpha*x0 + cut_sigma * client_eta
                                
                                # noisy_imgs cloud
                                cloud_noisy_imgs, cloud_log_snr, cloud_alpha, cloud_sigma = self.cloud.model(x_start=cut_noisy_imgs.cuda(),
                                                                                                            times=self.cloud.t,
                                                                                                            noise=cloud_eta.cuda(),
                                                                                                            noise_scheduler=self.cloud.noise_scheduler,
                                                                                                            random_crop_size=None) # shape: (batch_size,3,64,64)
                            else:
                                self.cloud.t = torch.tensor([]).long()
                    
                    # Getting model estimation of noise based on the images and the time-step
                    self.cloud.t_reshape = self.cloud.t.reshape(-1).long()
                    client.t_reshape = client.t.reshape(-1).long()                   
                    
                    LOGGER.debug(f't_cut: {client.t_cut_ratio}')
                    LOGGER.debug(f'client x0 shape: {x0.shape}')
                    LOGGER.debug(f'client t shape: {client.t.shape}')
                    LOGGER.debug(f'cloud t shape: {self.cloud.t.shape}')
                    LOGGER.debug(f'client t: {client.t}')
                    LOGGER.debug(f'cloud t: {self.cloud.t}')

                    #! Backward Process
                    #* Cloud Denoising (from t_cut to T)
                    if len(self.cloud.t) > 0:
                        match self.cloud.model_type:
                            case 'DDPM':
                                self.cloud.loss = self.cloud.model.backward(noisy_imgs=cloud_noisy_imgs.cuda(),
                                                                            imgs=x0.cuda(),
                                                                            t=self.cloud.t_reshape,
                                                                            noise=cloud_eta.cuda())
                            case 'IMAGEN':
                                self.cloud.loss = self.cloud.model.backward(x_noisy=cloud_noisy_imgs.cuda(),
                                                                            images=x0,
                                                                            times=self.cloud.t.cuda(),
                                                                            noise=cloud_eta.cuda(),
                                                                            log_snr=cloud_log_snr,
                                                                            alpha=cloud_alpha,
                                                                            sigma=cloud_sigma,
                                                                            noise_scheduler=self.cloud.noise_scheduler, 
                                                                            text_embeds=label_batch[0].cuda(),
                                                                            text_masks=label_batch[1].cuda())
                    else:
                        self.cloud.loss = None
                    # self.cloud.energy_usage['DENOISING_PROCESS'] = self.cloud.tracker.stop_task(task_name='DENOISING_PROCESS')
                    
                    #* Client Denoising (from zero to t_cut)
                    # client.tracker.start_task('DENOISING_PROCESS')
                    if len(client.t) > 0:
                        match self.cloud.model_type:
                            case 'DDPM':
                                client.loss = client.model.backward(noisy_imgs=client_noisy_imgs,
                                                                    imgs=x0,
                                                                    t=client.t_reshape,
                                                                    noise=client_eta.cuda())
                            case 'IMAGEN':
                                client.loss = client.model.backward(x_noisy=client_noisy_imgs.cuda(),
                                                                    images=x0,
                                                                    times=client.t.cuda(),
                                                                    noise=client_eta.cuda(),
                                                                    log_snr=client_log_snr,
                                                                    alpha=client_alpha,
                                                                    sigma=client_sigma,
                                                                    noise_scheduler=client.noise_scheduler, 
                                                                    text_embeds=label_batch[0].cuda(),
                                                                    text_masks=label_batch[1].cuda())
                    else:
                        client.loss = None

                    #* Cloud Update
                    if self.cloud.loss:
                        self.cloud.loss.backward()
                        self.cloud.optimizer.step()
                        self.cloud.optimizer.zero_grad()
                    
                    #* Client Update                 
                    if client.loss:
                        client.loss.backward()
                        client.optimizer.step()
                        client.optimizer.zero_grad()
                    
                    # Aggregate the loss of client and cloud model: lambda * client_loss + (1-lambda) * cloud_loss
                    if self.cloud.loss and client.loss:
                        loss = (self.loss_lambda * client.loss + (1-self.loss_lambda) * self.cloud.loss)
                    elif self.cloud.loss:
                        loss = self.cloud.loss
                    elif client.loss:
                        loss = client.loss
                    total_loss += loss.item()
                    self.step += 1

                # log metrics to wandb
                client_losses = {f'{c_id} loss':c_node.loss for c_id, c_node in self.clients.items()}
                wandb.log({"cloud loss": self.cloud.loss, 
                            "total loss": total_loss, 
                            **client_losses, 
                            })

            epoch_loss = total_loss

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
            
            if self.best_loss > epoch_loss:
                self.best_loss = epoch_loss
   
            if epoch % self.save_and_sample_every == 0:
                # Storing the model
                for client_id, client in self.clients.items():
                    self.save(model=client.model, 
                                optimizer=client.optimizer,
                                path_save_model=client.path_save_model)
                    
                    # wandb.log_artifact(artifact_or_path=client.path_save_model,name=client_id)
                        
                self.save(model=self.cloud.model,
                            optimizer=self.cloud.optimizer,
                            path_save_model=self.cloud.model.path_save_model) 
                
                # Display images generated at this epoch
                if self.display:
                    self.generate_images()
                
                # wandb.log_artifact(artifact_or_path=self.cloud.model.path_save_model,name=self.cloud.id)
                log_string += " --> Best model ever (stored)"
                LOGGER.info(log_string)

        for client_id, client in self.clients.items():
            self.save(model=client.model, 
                        optimizer=client.optimizer,
                        path_save_model=client.path_save_model)
        
            wandb.log_artifact(artifact_or_path=client.path_save_model, name=client_id)
            
        self.save(model=self.cloud.model,
                    optimizer=self.cloud.optimizer,
                    path_save_model=self.cloud.model.path_save_model)
        
        wandb.log_artifact(artifact_or_path=self.cloud.model.path_save_model,name=self.cloud.id)
        log_string += " --> Final model stored"
        
        self.generate_images()
        
        wandb.finish()

    def test(self):
        
        # 1. Start a W&B Run
        self.run = wandb.init(
            project="distributed_genai",
            notes="This experiment will trest distributed diffusion models to investigate the effectiveness regarding information inclosure, performance and resources",
            tags=["ecml", "experiment", "test", f"{self.cloud.id}"],
            name=f'test_{datetime.now().strftime("%I:%M%p_%m-%d-%Y")}_{self.cloud.dataset_name}_{self.cloud.id}_clients-{len(self.clients)}_s-{SETTINGS.imagen_model["DEFAULT"]["image_sizes"][0]}'
        )
        
        #  Capture a dictionary of the hyperparameters
        wandb.config['TRAINER'] = SETTINGS.diffusion_trainer
        wandb.config['DIFFUSION_MODEL'] = SETTINGS.diffusion_model
        wandb.config['UNET'] = SETTINGS.unet
        wandb.config['CLIENTS'] = SETTINGS.clients
        
        path_metric_test_results_folder = SETTINGS.diffusion_trainer['GENERATION']['path_metric_test_results_folder']
        
        self.load()
        self.generate_images(testing=True)
        
        test_results = pd.DataFrame(columns=['client_id', 't_cut', 'dir_name', 'fid', 'clip_fid', 'kid'])
        
        for dir_name in ['generated/', 'cut/', 'cloud_cut/', 'cloud_approx/', 'client_from_noise/']:
            
            for client_id, client in self.clients.items():
                if os.path.exists(self.results_folder + f"testing/{int(client.t_cut)}/{dir_name}"):
                    
                    fid_score, clip_fid_score, kid_score = client.compute_scores(results_folder=self.results_folder, dir_name=dir_name)
                    
                    LOGGER.info(f'Performance {client_id}-{dir_name.replace("/", "")} | FID: {fid_score} | CLIP_FID: {clip_fid_score} | KID: {kid_score}')
                    
                    row = pd.DataFrame({'client_id': client_id, 
                                't_cut': client.t_cut, 
                                'dir_name': dir_name, 
                                'fid': fid_score, 
                                'clip_fid': clip_fid_score, 
                                'kid': kid_score}, index=[0])
                    test_results = pd.concat([test_results, row], ignore_index=True)
            
            if os.path.exists(self.results_folder + f"testing/{int(client.t_cut)}/{dir_name}"):
                fid_score = fid.compute_fid(fdir1=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/{dir_name}/'), 
                                            fdir2=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/real/'), 
                                            mode="clean", 
                                            model_name="inception_v3")
            
                clip_fid_score = fid.compute_fid(fdir1=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/{dir_name}/'), 
                                            fdir2=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/real/'), 
                                            mode="clean", 
                                            model_name="clip_vit_b_32")
                
                kid_score = fid.compute_kid(fdir1=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/{dir_name}/'), 
                                            fdir2=os.path.join(self.results_folder, f'testing/{int(client.t_cut)}/real/'), 
                                            mode="clean")
                
                row = pd.DataFrame({'client_id': 'ALL', 
                                    't_cut': client.t_cut, 
                                    'dir_name': dir_name, 
                                    'fid': fid_score, 
                                    'clip_fid': clip_fid_score, 
                                    'kid': kid_score}, index=[0])
                test_results = pd.concat([test_results, row], ignore_index=True)
        
        wandb.log({"test_results": wandb.Table(dataframe=test_results)})  
        test_results.to_csv(path_metric_test_results_folder + f'test_results_{int(client.t_cut)}.csv', index=False)      
        
        visualize_performance(data=test_results, results_folder=path_metric_test_results_folder)
        
        wandb.finish()
    
    def generate_images(self, testing=False):
        
        # Obtain SETTINGS
        batch_size=SETTINGS.diffusion_trainer['DEFAULT']['batch_size']
        return_all_timesteps=SETTINGS.diffusion_trainer['GENERATION']['return_all_timesteps']
        n_samples=SETTINGS.diffusion_trainer['GENERATION']['n_samples']
        return_pil_images=True
        
        for batch_k in range(int(np.ceil(n_samples/batch_size))):
            
            LOGGER.info(f'Batch {batch_k} of {int(np.ceil(n_samples/batch_size))} batches')
            if testing:
                wandb.log({"batch_k": batch_k})
            
            if batch_k*batch_size < SETTINGS.diffusion_trainer['GENERATION']['start_sampling_from_idx']:
                continue
            
            if not SETTINGS.diffusion_trainer['GENERATION']['create_new_samples'] and testing:
                continue
            
            # Continue sampling using Clients
            for client_id, client in self.clients.items():
                
                img_batch, label_batch, text = next(iter(client.dl_test))
                
                noise_img = torch.randn_like(img_batch).cuda()
                
                # Sample using Clouds
                match self.cloud.model_type:
                    case 'DDPM':
                        # t cloud
                        self.cloud.t = torch.randint(client.t_cut,
                                                     self.cloud.model.num_timesteps,
                                                     (batch_size,), 
                                                     device=client.device).long()
                        # t cut
                        t_cut_tensor = torch.full_like(input=self.cloud.t, fill_value=client.t_cut)
                        # cut noisy_imgs
                        cut_noisy_imgs = self.cloud.model(x0=x0,
                                                          t=t_cut_tensor,
                                                          noise=client_eta.cuda(),
                                                          offset_noise_strength=self.offset_noise_strength)
                        if client.t_cut_ratio < 1:
                            cloud_img_samples = self.cloud.model.sample(batch_size=sample_batch_size,
                                                                        t_min=int(client.t_cut),
                                                                        t_max=self.cloud.model.num_timesteps,
                                                                        noise_img=noise_img,
                                                                        return_all_timesteps=False)
                            cloud_img_approx = self.cloud.model.sample(batch_size=sample_batch_size,
                                                                       t_min=0,
                                                                       t_max=int(client.t_cut),
                                                                       noise_img=cloud_img_samples.cuda(),
                                                                       return_all_timesteps=False)
                    case 'IMAGEN':
                        # t cloud
                        self.cloud.t = self.cloud.noise_scheduler.sample_random_times(batch_size, sample_interval=(client.t_cut_ratio,1),device = client.device) # shape: (batch_size)
                        # t cut
                        t_cut_tensor = torch.full_like(input=self.cloud.t, fill_value=client.t_cut_ratio)
                        
                        # cut noisy_imgs
                        log_snr = self.cloud.noise_scheduler.log_snr(t_cut_tensor).type(img_batch.dtype)
                        log_snr_padded_dim = func.right_pad_dims_to(img_batch, log_snr)
                        cut_alpha, cut_sigma =  func.log_snr_to_alpha_sigma(log_snr_padded_dim)
                        
                        cut_noisy_imgs = cut_alpha*img_batch.cuda() + cut_sigma * noise_img
                        
                        if client.t_cut_ratio < 1:
                            cloud_img_samples = self.cloud.model.sample(t_min=float(client.t_cut_ratio),
                                                                        t_max=1.0,
                                                                        n_sampling_timesteps=int((1-client.t_cut_ratio) * self.cloud.model.num_timesteps),
                                                                        noise_img=noise_img,
                                                                        text_embeds=label_batch[0].cuda(),
                                                                        text_masks=label_batch[1].cuda())
                            
                            cloud_img_approx = self.cloud.model.sample(t_min=0.0,
                                                                        t_max=float(client.t_cut_ratio),
                                                                        n_sampling_timesteps=int(float(client.t_cut_ratio) * self.cloud.model.num_timesteps),
                                                                        noise_img=cloud_img_samples.cuda(),
                                                                        text_embeds=label_batch[0].cuda(),
                                                                        text_masks=label_batch[1].cuda())
                            
                            LOGGER.info(f'{client.t_cut} - {cloud_img_samples.shape} shape')
                    
                if client.t_cut_ratio == 0:
                    img_samples=cloud_img_samples
                    client_img_samples=None
                    client_img_samples_from_noise=None
                    
                elif client.t_cut_ratio == 1:
                    cloud_img_samples=None
                    client_img_samples_from_noise=None
                    match client.model_type:
                        case 'DDPM':
                            client_img_samples = client.model.sample(batch_size=sample_batch_size,
                                                        t_min=0,
                                                        t_max=client.model.num_timesteps,
                                                        noise_img=noise_img.cuda(),
                                                        return_all_timesteps=False)
                        case 'IMAGEN':
                            client_img_samples = client.model.sample(t_min=0.0,
                                                        t_max=1.00,
                                                        n_sampling_timesteps=client.model.num_timesteps,
                                                        noise_img=noise_img.cuda(),
                                                        return_all_timesteps=return_all_timesteps,
                                                        text_embeds=label_batch[0].cuda(),
                                                        text_masks=label_batch[1].cuda())
                    # Store images
                    img_samples=client_img_samples
                    LOGGER.debug(f'CLIENT IMAGES {client_img_samples.shape} shape')
                            
                else:
                    match client.model_type:
                        case 'DDPM':
                            client_img_samples = client.model.sample(batch_size=sample_batch_size,
                                                        t_min=0,
                                                        t_max=int(client.t_cut + client.t_cut * (1-client.t_cut_ratio)),
                                                        noise_img=cloud_img_samples.cuda(),
                                                        return_all_timesteps=return_all_timesteps)
                            
                            client_img_samples_from_noise = client.model.sample(batch_size=sample_batch_size,
                                                                                t_min=0,
                                                                                t_max=client.model.num_timesteps,
                                                                                noise_img=noise_img.cuda(),
                                                                                return_all_timesteps=return_all_timesteps)
                        case 'IMAGEN':
                            client_img_samples = client.model.sample(t_min=0.0,
                                                                    t_max=float(client.t_cut_ratio) + float(client.t_cut_ratio * (1-client.t_cut_ratio)),
                                                                    n_sampling_timesteps=int(client.t_cut_ratio * client.model.num_timesteps),
                                                                    noise_img=cloud_img_samples.cuda(),
                                                                    text_embeds=label_batch[0].cuda(),
                                                                    text_masks=label_batch[1].cuda())
                            
                            client_img_samples_from_noise = client.model.sample(t_min=0.0,
                                                                                t_max=1.0,
                                                                                n_sampling_timesteps=int(client.t_cut_ratio * client.model.num_timesteps),
                                                                                noise_img=noise_img.cuda(),
                                                                                text_embeds=label_batch[0].cuda(),
                                                                                text_masks=label_batch[1].cuda())
                            
                            LOGGER.info(f'CLIENT IMAGES {client_img_samples.shape} shape')
                    
                    img_samples=client_img_samples
                
                #* Client images
                for batch_idx, imgs in enumerate(img_samples):
                    
                    image_idx=batch_k*batch_size + batch_idx
                    imgs_raw=(func.unnormalize_to_zero_to_one(imgs.cpu().numpy().squeeze()) * 255).astype(np.uint8)
                    if testing:
                        imgs_orig=(func.unnormalize_to_zero_to_one(img_batch[batch_idx].cpu().numpy().squeeze()) * 255).astype(np.uint8)
                        imgs_cut=(func.unnormalize_to_zero_to_one(cut_noisy_imgs[batch_idx].cpu().numpy().squeeze()) * 255).astype(np.uint8)
                        if cloud_img_samples is not None:
                            cloud_cut=(func.unnormalize_to_zero_to_one(cloud_img_samples[batch_idx].cpu().numpy().squeeze()) * 255).astype(np.uint8)
                            cloud_approx=(func.unnormalize_to_zero_to_one(cloud_img_approx[batch_idx].cpu().numpy().squeeze()) * 255).astype(np.uint8)
                        if client_img_samples is not None and client.t_cut_ratio < 1:
                            client_img_from_noise=(func.unnormalize_to_zero_to_one(client_img_samples_from_noise[batch_idx].cpu().numpy().squeeze()) * 255).astype(np.uint8)
                    
                    if testing:
                        folder=self.results_folder + f"testing/{int(client.t_cut)}/"
                        if not os.path.exists(folder):
                            os.mkdir(folder)
                            
                        for dir_name in ['real/', 'generated/', 'cut/']:
                            if not os.path.exists(folder + dir_name):
                                os.mkdir(folder + dir_name)
                            if not os.path.exists(folder + dir_name + f"{client_id}/"):
                                os.mkdir(folder + dir_name + f"{client_id}/")
                                
                        image_save_path=os.path.join(folder, "generated", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                        image_orig_save_path=os.path.join(folder, "real", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                        image_cut_save_path=os.path.join(folder, "cut", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                        
                        if cloud_img_samples is not None:
                            for dir_name in ['cloud_cut/', 'cloud_approx/']:
                                if not os.path.exists(folder + dir_name):
                                    os.mkdir(folder + dir_name)
                                if not os.path.exists(folder + dir_name + f"{client_id}/"):
                                    os.mkdir(folder + dir_name + f"{client_id}/")
                              
                            image_cloud_cut_save_path=os.path.join(folder, "cloud_cut", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                            image_cloud_approx_save_path=os.path.join(folder, "cloud_approx", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                            
                        if client_img_samples is not None:
                            for dir_name in ['client_from_noise/']:
                                if not os.path.exists(folder + dir_name):
                                    os.mkdir(folder + dir_name)
                                if not os.path.exists(folder + dir_name + f"{client_id}/"):
                                    os.mkdir(folder + dir_name + f"{client_id}/")
                                
                            image_client_from_noise_save_path=os.path.join(folder, "client_from_noise", f"{client_id}", f"image_{client_id}_{image_idx}_{text[batch_idx]}.png")
                    else:
                        image_save_path=self.results_folder + f"training/image_{int(client.t_cut)}_{client_id}_{image_idx}_{text[batch_idx]}.png"

                    # generated images
                    img_raw=imgs_raw.transpose(1, 2, 0)
                    
                    if SETTINGS.diffusion_trainer['GENERATION']['save_train_samples'] or testing:
                        img = Image.fromarray(img_raw)
                        img.save(image_save_path)
                    
                    if testing:
                        # original images
                        img_orig=imgs_orig.transpose(1, 2, 0)
                        img = Image.fromarray(img_orig)
                        img.save(image_orig_save_path)  
                        
                        img_cut=imgs_cut.transpose(1, 2, 0)
                        img = Image.fromarray(img_cut)
                        img.save(image_cut_save_path) 
                        
                        if cloud_img_samples is not None:
                            cloud_cut=cloud_cut.transpose(1, 2, 0)
                            img = Image.fromarray(cloud_cut)
                            img.save(image_cloud_cut_save_path)
                            
                            cloud_approx=cloud_approx.transpose(1, 2, 0)
                            img = Image.fromarray(cloud_approx)
                            img.save(image_cloud_approx_save_path)
                        
                        if client_img_samples_from_noise is not None and client.t_cut_ratio < 1:
                            client_img_from_noise=client_img_from_noise.transpose(1, 2, 0)
                            img = Image.fromarray(client_img_from_noise)
                            img.save(image_client_from_noise_save_path)
                    
                    else:
                        wandb.log({f'Gen_Img_e{self.epoch}-{client_id}_{text[batch_idx]}': wandb.Image(img_raw)})

                        #! Caution
                        break
                         
            if not testing:
                break