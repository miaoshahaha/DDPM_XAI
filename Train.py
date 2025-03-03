import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Model import GaussianDiffusionSampler, GaussianDiffusionTrainer, GradualWarmupScheduler, UNet
from Dataset import CUSTOM_DATASET

def check_reduce():
    """
    Determines whether to use the reduced dataset.
    Returns:
        bool: True or False
    """
    return True

def create_reduced_data(args, reduced_sort_idx, modelConfig):
    """
    Reduces the training dataset by index selected.
    Args:
        args: Function arguments.
        reduced_sort_idx: Sorted indices of samples based on importance.
        modelConfig: Dictionary containing model configuration.

    Returns:
        Subset: A reduced training dataset.
    """
    reduced_percentile = 0.1

    dt_train = CUSTOM_DATASET(args, split=True)
    train_dataset, _ = dt_train.load_dataset(custom_trasform=True)
    HIHE_reduced_data = Subset(train_dataset, list(reduced_sort_idx[int(len(reduced_sort_idx)*reduced_percentile):]))

    return HIHE_reduced_data
    

def train_new(args, reduced_sort_idx, modelConfig: Dict):
    """
    Trains a new DDPM model using the specified datasets and config.
    Args:
        args: Function arguments.
        reduced_sort_idx: Sorted indices of samples based on importance.
        modelConfig: Dictionary containing model configuration.

    """
    device = torch.device(modelConfig["device"])
    # dataset
    
    dt = CUSTOM_DATASET(args)
    train_dataset, test_dataset = dt.load_dataset()

    #Check if reducing the data
    if check_reduce():
        train_dataset = create_reduced_data(args, reduced_sort_idx, modelConfig)
    
    dataloader = DataLoader(
        train_dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(1, modelConfig["epoch"]+1):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        
        #if modelConfig["save_weight_dir"][2:-1] not in os.listdir(os.path.dirname(__file__)): # Only characters
            #os.mkdir(modelConfig["save_weight_dir"])

        """Modify in different enviornment"""
        if e > modelConfig["epoch"] - 5 or e % 10 == 0:
            torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], f'rd{int(args.rd * 100)}_c{args.split_class[0]}_ckpt_{e}_.pt'))


def evaluate(args, modelConfig: Dict):
    """
    Load a trained model and generates images using Gaussian diffusion sampling.
    Args:
        args: Function arguments.
        modelConfig: Dictionary containing model configuration.

    """
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

        """Modify in different enviornment"""
        ckpt = torch.load(os.path.join(
            modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        
        #if modelConfig["sampled_dir"][2:-1] not in os.listdir(os.path.dirname(__file__)): # Only characters
            #os.mkdir(modelConfig["sampled_dir"])

        """Modify in different enviornment"""
        save_image(saveNoisy, os.path.join(
            modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        
        """Modify in different enviornment"""
        save_image(sampledImgs, os.path.join(
            modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])