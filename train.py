import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from model.unet.ConvUNet import *
from utilities import *

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg:DictConfig):

    DS = BaseDataset(**cfg.dataset)
    train_DL, test_DL = prepare_data.prepare_dataloader(DS, cfg.train_params.batch_size)

    if cfg.check_dataset:
        prepare_data.check_dataset(DS)

    # Load model
    model = ConvUNet()

    # Train
    trainer = L.Trainer(max_epochs=cfg.epochs) #callbacks=[LearningRateMonitor(logging_interval='epoch'), TrainLossPlotter('output')]
    logger = TensorBoardLogger('outputs', name='SR')
    trainer.fit(model, train_DL, test_DL, logger=logger)

    '''train_loss = trainer.callback_metrics['train_loss']
    val_loss = trainer.callback_metrics['val_loss']'''

    # Plot the train and validation loss curves
    # curves_plot(train_loss, val_loss)
if __name__ == '__main__':
    main()