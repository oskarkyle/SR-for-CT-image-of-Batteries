import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from clearml import Task

from model.unet.ConvUNet import *
from source.BaseDataset import *
from source.Check import *

def custom_collate(batch):
    """
    Custom collate function to drop none informative tiles on edge to prevent model params pollution
    """
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

def init_model(cfg:DictConfig):
    channels = cfg.image_channels
    c_factor = cfg.c_factor
    output_channels = cfg.output_channels
    ch_mults = cfg.ch_mults
    n_blocks = cfg.n_blocks
    n_layers = cfg.n_layers
    scale_factor = cfg.scale_factor
    kernel_size = cfg.kernel_size
    n_groups = cfg.n_groups
    verbose = cfg.verbose
    model = ConvUNet(cfg,channels,output_channels,c_factor,ch_mults,n_blocks,n_layers,scale_factor,kernel_size,n_groups,verbose)

    return model


def setup_dataloaders(cfg:DictConfig):
    dataset = BaseDataset(**cfg.dataset)
    split_ratios = cfg.split_ratios
    train_ratio = split_ratios.train
    val_ratio = split_ratios.val 
    test_ratio = split_ratios.test

    dataset_size = dataset.__len__()
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    train_dataset,validation_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])

    train_dataloader = DataLoader(train_dataset, batch_size = cfg.train.batch_size, shuffle = True,collate_fn= custom_collate,pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = cfg.train.batch_size, shuffle = False,collate_fn= custom_collate,pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size = cfg.train.batch_size, shuffle = False,collate_fn= custom_collate,pin_memory=True)
    return train_dataloader,validation_dataloader,test_dataloader

def setup_logger(cfg:DictConfig):
    log_dir = os.getcwd() + cfg.train.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_name = cfg.model.model_name
    logger = TensorBoardLogger(save_dir = log_dir, name = model_name)
    return logger


def start_training(cfg: DictConfig):
    model_cfg = cfg.model
    train_cfg = cfg.train
    myckpt_dir = os.getcwd() + train_cfg.save_dir

    if not os.path.exists(myckpt_dir):
        os.mkdir(myckpt_dir)

    myckpt_path = os.getcwd() + train_cfg.save_dir + train_cfg.save_name +  '.ckpt'
    oldckpt_path = os.getcwd() + cfg.pred.ckpt_path
    checkpoint_callback = ModelCheckpoint(
        monitor = train_cfg.callbacks.monitor, 
        dirpath = os.getcwd() + train_cfg.save_dir,  
        filename = train_cfg.save_name,  
        save_top_k = 1,  
        mode = train_cfg.callbacks.mode        
    )

    early_stop_callback = (
        EarlyStopping(
            monitor = train_cfg.callbacks.monitor ,
            patience =train_cfg.callbacks.patience,
            mode = train_cfg.callbacks.mode
        )
    )

    train_dataloader,validation_dataloader,test_dataloager = setup_dataloaders(cfg)

    if os.path.isfile(oldckpt_path):
        logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {ConvUNet.__name__}\n\tcheckpoint: {oldckpt_path}")
        model = ConvUNet.load_from_checkpoint(oldckpt_path,
                                            map_location=torch.device('mps'),
                                            ch_mults=model_cfg.ch_mults,
                                            n_blocks=model_cfg.n_blocks,
                                            n_layers=model_cfg.n_layers
        )
        logger.success(f"Successfully loaded checkpoint")

    else:
        print("No checkpoint found, initializing new model")
        model = init_model(model_cfg)
    print(myckpt_path)
    trainer = L.Trainer(max_epochs=train_cfg.epochs,
                         callbacks=[checkpoint_callback,early_stop_callback],
                         precision=train_cfg.precision , 
                         accelerator=train_cfg.accelerator,
                         logger = setup_logger(cfg),
                         log_every_n_steps=1)
    
    if os.path.exists(myckpt_path):
        trainer.fit(model, train_dataloader,validation_dataloader, myckpt_path)
    else:
        trainer.fit(model, train_dataloader,validation_dataloader)