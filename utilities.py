import os 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import Callback


from source.BaseDataset import *
from source.data_utils import *

def data(cfg: DictConfig, preprocess_cfgs: DictConfig, transform_cfgs: DictConfig = None):
    config = OmegaConf.load(cfg)
    preprocess_cfgs = OmegaConf.load(preprocess_cfgs)

    data_root = config.data_params.data_root
    dataset_dir = config.data_params.dataset_dir
    tile_grid = config.data_params.tile_grid
    size = config.data_params.tile_size
    batch_size = config.train_params.batch_size

    dataset = BaseDataset('SR', 'train', size, tile_grid, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
    train_dataloader, test_dataloader = prepare_data.prepare_dataloader(dataset, batch_size)

    return dataset, train_dataloader, test_dataloader


def curves_plot(train_loss: L.Trainer.callback_metrics, val_loss: L.Trainer.callback_metrics):
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

def pred_data(cfg: DictConfig, preprocess_cfgs: DictConfig, transform_cfgs: DictConfig = None):
    config = OmegaConf.load(cfg)
    preprocess_cfgs = OmegaConf.load(preprocess_cfgs)

    data_root = config.data_params.data_root
    dataset_dir = config.data_params.dataset_dir
    tile_grid = config.data_params.tile_grid
    size = config.data_params.tile_size
    batch_size = config.train_params.batch_size

    dataset = BaseDataset('SR', 'pred', size, tile_grid, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    subset = torch.utils.data.Subset(dataset, subset_indices)
    pred_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    return subset, pred_dataloader


def is_lightning_module(model):
    return isinstance(model, L.LightningModule)


def plot_img(predictions, pred_dataloader):
    pred = []
    for i, batch in enumerate(predictions):
        for j, img in enumerate(batch):
            print(img.shape)
            img = img.squeeze().numpy()
            pred.append(img)

    input, label = prepare_data.check_dataloader(pred_dataloader)
    
    vmin = 0.2
    vmax = 0.9
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
            plt.savefig(f'./results/simple_{i+1}.png')
            plt.show()



if __name__ =='__main__':
    cfg = r'.\configs\train.yaml'
    preprocess_cfgs = r'.\configs\preprocess.yaml'

    dataset, train_dataloader, test_dataloader = data(cfg, preprocess_cfgs)

    prepare_data.check_dataset(dataset)