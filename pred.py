import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import lightning as L

from source.BaseDataset import BaseDataset
from source.data_utils import *
from utilities import *
from model.unet.ConvUNet import *

@hydra.main(config_path="configs", config_name="pred")
def main(cfg:DictConfig):
    print(cfg)
    print(cfg.dataset)

    DS = BaseDataset(**cfg.dataset)
    
    subset_indices = cfg.pred.subset_indices
    batch_size = cfg.pred.batch_size

    pred_DS = torch.utils.data.Subset(DS, subset_indices)
    pred_DL = DataLoader(pred_DS, batch_size)

    # prepare_data.check_dataset(pred_DS)
    # Load model

    ckpt_path = cfg.model.ckp_path

    model: L.LightningModule = ConvUNet().restore(ckpt_path)
    print(is_lightning_module(model))


    # Predict
    trainer = L.Trainer()
    predictions = trainer.predict(model, pred_DL)

    # Plot
    plot_img(predictions, pred_DL)

if __name__ == "__main__":
    main()
