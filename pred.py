import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning import Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset
from PIL import Image

from source.BaseDataset import *
from utilities import *
from model.unet.ConvUNet import *
from train import *

def custom_collate(batch):
    """
    Custom collate function to drop none informative tiles on edge to prevent model params pollution
    """
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

def load_model(cfg: DictConfig):
    path = cfg.dataset.data_root
    os.chdir(path)
    myckpt_path = os.getcwd() + cfg.pred.ckpt_path
    logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {ConvUNet.__name__}\n\tcheckpoint: {myckpt_path}")
    model = ConvUNet.load_from_checkpoint(myckpt_path, map_location=torch.device('mps'), ch_mults=cfg.model.ch_mults, n_blocks=cfg.model.n_blocks, n_layers=cfg.model.n_layers)
    logger.success(f"Successfully loaded checkpoint")

    return model

def setup_pred_tiles_and_labels(cfg: DictConfig):
    tiles = []
    labels = []
    pred_DS = BaseDataset(**cfg.dataset)
    subset_indices = list(range(0, cfg.pred.length))
    pred_DS = Subset(pred_DS, subset_indices)

    for i in range(len(pred_DS)):
        input, label = pred_DS.__getitem__(i)
        
        tiles.append(input)
        labels.append(label)
    return tiles, labels

def prediction(model, tile):
    device = torch.device("mps")
    input = tile.to(device)
    output = model(input)
    return output

def inference(cfg: DictConfig):
    model = load_model(cfg)
    tiles, labels = setup_pred_tiles_and_labels(cfg)
    for i in range(len(tiles)):
        tile = tiles[i].unsqueeze(0)
        output = prediction(model, tile)
        tile = tile.cpu().squeeze(0)
        input = tile.squeeze(0)
        label = labels[i].squeeze(0)    
        output = output.squeeze(0).cpu().detach().numpy()
        pred = output.squeeze(0)
        output_path = cfg.dataset.data_root + cfg.pred.output_dir + cfg.pred.output_name + f'_{i}.png'
        save_image(input, pred, label, output_path)
