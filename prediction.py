import argparse
import os

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torch.nn.functional as F
from tqdm import tqdm

from BaseDataset import *
from model.SRCNN import *

def evaluate_mse(created_images, ground_truth_images):
    mse = F.mse_loss(created_images, ground_truth_images, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])
    return mse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', dest='size', type = int, help='The size of each tile in pages in tiff', default=256)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32)
args = parser.parse_args()

# Prepare dataset
# Load data
data_root = '/Users/haoruilong/BA_code/SR_for_CT_image_of_Batteries'
dataset_dir = ['/Dataset/Pristine']

# Prepare configurations for dataset
cfgs_path_p = data_root + '/configs/preprocess.yaml'
cfgs_path_t = data_root + '/configs/transform.yaml'

if os.path.exists(cfgs_path_p):
    preprocess_cfgs = OmegaConf.load(cfgs_path_p)
else:
    preprocess_cfgs = None

if os.path.exists(cfgs_path_t):
    transform_cfgs = OmegaConf.load(cfgs_path_t)
else:
    preprocess_cfgs = None

# Dataset for all, default size: 256*256
mydataset = BaseDataset('SR', 'eval', args.size, dataset_dir, data_root, None, preprocess_cfgs)

# Data splitting
length_dataset = mydataset.__len__()
train_size = int(0.8 * length_dataset)
test_size = length_dataset - train_size

train_dataset, test_dataset = data_utils.random_split(mydataset, [train_size, test_size])
# Apply train and test dataset in dataloader
mydataloader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# Load the trained model 
device = torch.device("mps")
model = SRCNN()
model.to(device)
state_dict = torch.load('./outputs/epoch_99.pth', map_location=device)
model.load_state_dict(state_dict)
logger.info(f"model loaded")

# Test if test_dataloader successfully load the dataset
"""
logger.info(f"check the dataloader...")
for batch in test_dataloader:
    inputs, labels = batch
    print(inputs.shape)
    pass
logger.info(f"dataset loaded")
"""

# Evaluate the created images by a trained model against the ground truth
min_mse = float('inf')
for batch in test_dataloader:
    inputs, labels = batch
    
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds = model(inputs)

    num_images_inputs = inputs.shape[0]
    num_images_preds = preds.shape[0]
    if num_images_inputs == num_images_preds:
        num_images = num_images_preds
        print(num_images)
        
    for i in range(num_images):
        input = inputs[i]
        label = labels[i]
        pred = preds[i]
        
        input = input.squeeze().cpu().numpy()
        label = label.squeeze().cpu().numpy()
        pred = pred.detach().squeeze().cpu().numpy()


        fig, axes = plt.subplots(1, 3)#, figsize=(10, 8)

        axes[0].imshow(input, cmap='gray')
        axes[0].set_title('input')

        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('pred')

        axes[2].imshow(label, cmap='gray')
        axes[2].set_title('label')

        plt.tight_layout(pad=0.5)
        plt.show()

    """
    mse = evaluate_mse(inputs, labels)
    mean_mse = mse.mean().item()  # Average MSE across the batch
    print(f"Mean Squared Error: {mean_mse}")
    # Choose the lowest mse in loop
    if mean_mse < min_mse:
        min_mse = mean_mse
    """

# print(f"Lowest Mean Squared Error: {min_mse}")