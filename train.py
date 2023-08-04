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

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--tile_grid', dest='tile_grid', type=int, default=4, help='The number of tiles in each row/column in tiff')
parser.add_argument('-s', '--size', dest='size', type = int, help='The size of each tile in pages in tiff', default=512)
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=1e-4)
parser.add_argument('-ep', '--num_epochs', dest='num_epochs', type=int, default=100)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4)
parser.add_argument('-bi', '--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
args = parser.parse_args()


class TrainLossPlotter(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.losses.append(trainer.callback_metrics['train_loss'].item())

    def on_train_end(self, trainer, pl_module):
        plt.plot(range(1, trainer.current_epoch + 2), self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, 'train_loss_curve.png'))
        plt.show()




if __name__ == '__main__':
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


    dataset = prepare_data.prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, args.size, args.tile_grid)
    train_dataloader, test_dataloader = prepare_data.prepare_dataloader(dataset, args.batch_size)

    #prepare_data.check_dataset(dataset)

    model = ConvUNet()

    trainer = L.Trainer(max_epochs=1) #callbacks=[LearningRateMonitor(logging_interval='epoch'), TrainLossPlotter('output')]
    
    trainer.fit(model, train_dataloader, test_dataloader)

    train_loss = trainer.callback_metrics['train_loss']
    val_loss = trainer.callback_metrics['val_loss']

    # Plot the train and validation loss curves
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()