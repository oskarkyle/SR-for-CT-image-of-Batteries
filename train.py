from omegaconf import DictConfig, ListConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor


from source.BaseDataset import BaseDataset
from model.unet.ConvUNet import *
from source.data_utils import *
from utilities import *

@hydra.main(config_path="configs", config_name="train")
def main(cfg:DictConfig):
    print(cfg)
    print(cfg.dataset)

    DS = BaseDataset(**cfg.dataset)
    train_DL, test_DL = prepare_data.prepare_dataloader(DS, cfg.train_params.batch_size)

    prepare_data.check_dataset(DS)

    # Load model
    model = ConvUNet()

    # Train
    trainer = L.Trainer(max_epochs=1) #callbacks=[LearningRateMonitor(logging_interval='epoch'), TrainLossPlotter('output')]
    
    trainer.fit(model, train_DL, test_DL)

    train_loss = trainer.callback_metrics['train_loss']
    val_loss = trainer.callback_metrics['val_loss']

    # Plot the train and validation loss curves
    curves_plot(train_loss, val_loss)

if __name__ == '__main__':
    main()
