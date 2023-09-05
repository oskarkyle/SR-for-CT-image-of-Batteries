import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
import torchvision.transforms as transforms
import time
from loguru import logger
import os


def disassemble_dataloader(dataloader):
    input = []
    label = []

    for data in dataloader:
        inputs, labels = data
        for img in inputs:
            img = transforms.ToPILImage()(img)
            input.append(img)

        for img in labels:
            img = transforms.ToPILImage()(img)
            label.append(img)

    return input, label

# To plot the images from the prediction
def plot(predictions: list, pred_dataloader: DataLoader):
    pred = []
    for img_batch in predictions:
        for img in img_batch:
            img = img.squeeze(0)
            img_array = img.numpy()

            pred.append(img_array)

    input, label = disassemble_dataloader(pred_dataloader)
    
    vmin = 0
    vmax = 1
    
    if len(pred) == len(input):
        for i in range(len(pred)):
            
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

            ax[0].imshow(input[i], cmap='gray')
            title = ax[0].set_title('Input')
            title.set_fontsize(20)
            ax[1].imshow(pred[i], cmap='gray', vmin = vmin, vmax = vmax)
            title = ax[1].set_title('Prediction')
            title.set_fontsize(20)
            ax[2].imshow(label[i], cmap='gray')
            title = ax[2].set_title('Label')
            title.set_fontsize(20)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'./results/predictions_{i}.png')

def save_image(input, pred, label, output_path):
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