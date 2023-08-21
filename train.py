import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from model.unet.ConvUNet import *
from utilities import *

from clearml import Task

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg:DictConfig):
    ############################################################################################################
    task = Task.init(project_name="Students",
                     task_name=f"Super Resolution",
                     task_type=Task.TaskTypes.training,
                     reuse_last_task_id=False,
                     auto_resource_monitoring=False,
                     auto_connect_frameworks={"pytorch": False} # does not upload all the output models
                     )
    ############################################################################################################

    DS = BaseDataset(**cfg.dataset)
    train_DL, test_DL = prepare_data.prepare_dataloader(DS, cfg.train_params.batch_size)

    if cfg.check_dataset:
        prepare_data.check_dataset(DS)

    # Load model
    model = ConvUNet()

    # Train
    logger = TensorBoardLogger('outputs', name='SR', default_hp_metric=False)
    trainer = L.Trainer(max_epochs=cfg.epochs, logger=logger) #callbacks=[LearningRateMonitor(logging_interval='epoch'), TrainLossPlotter('output')]

    trainer.fit(model, train_DL, test_DL)

    # train_loss = trainer.callback_metrics['train_loss']
    # val_loss = trainer.callback_metrics['val_loss']

    # Plot the train and validation loss curves
    # curves_plot(train_loss, val_loss)
if __name__ == '__main__':
    # Plot curves with tensorboard
    main()