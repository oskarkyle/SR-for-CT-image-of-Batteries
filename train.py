import argparse
import os
import copy
import numpy as np

import torch
from torch import nn, tensor
import torch.optim as optim
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor

import matplotlib.pyplot as plt

from model.unet.ConvUNet import *
from source.data_utils import *
from utilities import *


if __name__ == '__main__':
    # config's path
    train_cfg = r'.\configs\train.yaml'
    preprocess_cfgs = r'.\configs\preprocess.yaml'


    # Data
    dataset, train_dataloader, test_dataloader = data(train_cfg, preprocess_cfgs)

    # Load model
    model = ConvUNet()

    # Train
    trainer = L.Trainer(max_epochs=1) #callbacks=[LearningRateMonitor(logging_interval='epoch'), TrainLossPlotter('output')]
    
    trainer.fit(model, train_dataloader, test_dataloader)

    train_loss = trainer.callback_metrics['train_loss']
    val_loss = trainer.callback_metrics['val_loss']

    # Plot the train and validation loss curves
    curves_plot(train_loss, val_loss)