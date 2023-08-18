import os 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import Callback


from source.BaseDataset import *
from source.data_utils import *

def curves_plot(train_loss: L.Trainer.callback_metrics, val_loss: L.Trainer.callback_metrics):
    # Plot the train and validation loss curves
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, vmin=0, vmax=1, label='Train Loss')
    plt.plot(epochs, val_loss, vmin=0, vmax=1, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.title('Train and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/loss_curves.png')
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


def is_lightning_module(model):
    return isinstance(model, L.LightningModule)

# To plot the images from the prediction
def plot_pred_imgs(predictions, pred_dataloader):
    pred = []
    for i, batch in enumerate(predictions):
        for j, img in enumerate(batch):
            print(img.shape)
            img = img.squeeze().numpy()
            pred.append(img)

    input, label = prepare_data.check_dataloader(pred_dataloader)
    
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
            plt.savefig(f'./results/pred_{i}.png')