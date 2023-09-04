import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning import Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset


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
    model_cfg = cfg.model
    myckpt_path = os.getcwd() + cfg.pred.ckpt_path

    model,config = ConvUNet(model_cfg= model_cfg,
                                    image_channels=model_cfg.image_channels, 
                                    output_channels=model_cfg.output_channels,
                                    c_factor=model_cfg.c_factor,
                                    ch_mults=model_cfg.ch_mults,
                                    n_blocks=model_cfg.n_blocks,
                                    n_layers=model_cfg.n_layers,
                                    scale_factor=model_cfg.scale_factor,
                                    kernel_size=model_cfg.kernel_size,
                                    n_groups=model_cfg.n_groups,
                                    verbose=model_cfg.verbose
                                    
                                    ).restore(myckpt_path, map_location=cfg.pred.device)
    return model

def setup_dataloader(cfg: DictConfig):
    pred_DS = Pred_dataset(**cfg.pred.dataset)
    subset_indices = list(range(0, cfg.length + 1))
    pred_DS = Subset(pred_DS, subset_indices)
    pred_dataloader = DataLoader(pred_DS, batch_size=cfg.pred.batch_size, shuffle=True, collate_fn=custom_collate,pin_memory=True)
    return pred_dataloader

def start_prediction(cfg: DictConfig):
    model = load_model(cfg)
    pred_dl = setup_dataloader(cfg)
    model.eval()

    with torch.no_grad():
        outputs = model(pred_dl)
    plot(outputs,pred_dl)

@hydra.main(version_base=None, config_path="configs", config_name="pred")
def main(cfg:DictConfig):
    load_model(cfg)

if __name__ == "__main__":
    main()
