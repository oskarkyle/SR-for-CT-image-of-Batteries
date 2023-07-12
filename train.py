import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import pytorch_lightning as L

from unet.ConvUNet import ConvUNet
from data_utils import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', type=str, default='SRCNN')
parser.add_argument('-s', '--size', dest='size', type = int, help='The size of each tile in pages in tiff', default=512)
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=1e-4)
parser.add_argument('-ep', '--num_epochs', dest='num_epochs', type=int, default=100)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('-bi', '--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
args = parser.parse_args()

# Prepare dataloader for training and testing
# Data path
data_root = f'H:\SR_for_CT_image_of_Batteries'
dataset_dir = [f'\dataset\pristine']

# Prepare configurations for dataset
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

if __name__ == '__main__':
    dataset = prepare_data.prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, args.size)
    train_dataloader, test_dataloader = prepare_data.prepare_dataloader(dataset, args.batch_size)

    model = ConvUNet(image_channels=1, output_channels=1)
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader, test_dataloader)
