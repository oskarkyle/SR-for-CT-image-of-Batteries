import matplotlib.pyplot as plt
from torch.utils.data import random_split
from natsort import natsorted
from loguru import logger
import cv2
from source.BaseDataset import *
from image_utils.utils import *
import numpy as np

def split_dataset(cfg: DictConfig, dataset: BaseDataset):
    split_ratios = cfg.split_ratios
    train_ratio = split_ratios.train
    val_ratio = split_ratios.val 
    test_ratio = split_ratios.test

    dataset_size = dataset.__len__()
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    train_dataset,validation_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])
    return train_dataset,validation_dataset,test_dataset

def save_image(input: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, output_path):
    input = convert_tensor_to_numpy(input)
    pred = convert_tensor_to_numpy(pred)
    label = convert_tensor_to_numpy(label)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(input, cmap='gray')
    title = ax[0].set_title('Input')
    title.set_fontsize(20)

    ax[1].imshow(pred, cmap='gray')
    title = ax[1].set_title('Prediction')
    title.set_fontsize(20)

    ax[2].imshow(label, cmap='gray')
    title = ax[2].set_title('Label')
    title.set_fontsize(20)

    
    plt.tight_layout()
    plt.savefig(output_path)

def save_image_in_RGB(input: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, output_path: str):
    input = convert_tensor_to_numpy(input)
    pred = convert_tensor_to_numpy(pred)
    label = convert_tensor_to_numpy(label)

    input_rgb = convert_gray_to_rgb(input)
    pred_rgb = convert_gray_to_rgb(pred)
    label_rgb = convert_gray_to_rgb(label)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(input_rgb)
    title = ax[0].set_title('Input')
    title.set_fontsize(20)

    ax[1].imshow(pred_rgb)
    title = ax[1].set_title('Prediction')
    title.set_fontsize(20)

    ax[2].imshow(label_rgb)
    title = ax[2].set_title('Label')
    title.set_fontsize(20)

    
    plt.tight_layout()
    plt.savefig(output_path)

def setup_pred_tiles_and_labels(dataset: BaseDataset):
    tiles = []
    labels = []

    for i in range(dataset.__len__()):
        input, label = dataset.__getitem__(i)
        tiles.append(input)
        labels.append(label)
    
    return tiles, labels

def calc_total_page_of_one_file(cfg: DictConfig, dataset: BaseDataset):
    total_tiles_in_one_page = cfg.dataset.tile_grid ** 2
    file_list = BaseDataset.load_file_paths_from_dir(cfg.dataset.data_root, cfg.dataset.dataset_dir)
    total_files = len(file_list)
    
    dataset_size = dataset.__len__()
    total_page = dataset_size // total_tiles_in_one_page
    total_page_of_one_file = total_page // total_files
    return total_page_of_one_file