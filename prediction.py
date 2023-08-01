import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import pytorch_lightning as L
from omegaconf import DictConfig, ListConfig, OmegaConf
from loguru import logger
import matplotlib.pyplot as plt
import time

from torchvision import transforms
from typing import Any, Tuple, Union, List, Sequence

from BaseDataset import BaseDataset
from data_utils import prepare_data
from unet.ConvUNet import *

def is_lightning_module(model):
    return isinstance(model, L.LightningModule)

def prediction(model: L.LightningModule, dataloader):
    trainer = L.Trainer()
    predictions = trainer.predict(model, dataloader)
    return predictions

def plot_img(predictions, pred_dataloader):
    pred = []
    for i, batch in enumerate(predictions):
        for j, img in enumerate(batch):
            print(img.shape)
            img = img.squeeze().numpy()
            pred.append(img)

    input, label = prepare_data.check_dataloader(pred_dataloader)

    if len(pred) == len(input):
        for i in range(len(pred)):
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(input[i], cmap='gray')
            ax[0].set_title('Input')
            ax[1].imshow(pred[i], cmap='gray')
            ax[1].set_title('Prediction')
            ax[2].imshow(label[i], cmap='gray')
            ax[2].set_title('Label')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    data_root = r'H:\SR_for_CT_image_of_Batteries'
    dataset_dir = [r'\dataset\pristine']

    cfgs_path_p = data_root + '\configs\preprocess.yaml'
    cfgs_path_t = data_root + '\configs\transform.yaml'

    if os.path.exists(cfgs_path_p):
        preprocess_cfgs = OmegaConf.load(cfgs_path_p)
    else:
        preprocess_cfgs = None

    if os.path.exists(cfgs_path_t):
        transform_cfgs = OmegaConf.load(cfgs_path_t)
    else:
        transform_cfgs = None

    batch_size = 4
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    dataset = prepare_data.prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, 512, 4)
    pred_dataset = prepare_data.prepare_pred_dataset(dataset, subset_indices)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

    # check dataset
    # prepare_data.check_dataset(pred_dataset)

    ckpt_path = r'.\lightning_logs\version_5\checkpoints\epoch=0-step=9280.ckpt'
    print(is_lightning_module(ConvUNet()))

    model: L.LightningModule = ConvUNet().restore(ckpt_path)
    print(is_lightning_module(model))

    predictions = prediction(model, pred_dataloader)

    plot_img(predictions, pred_dataloader)
    