from typing import Union, Tuple, Sequence
import importlib
import os

import lightning as L

import torch
import torchmetrics
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer

###from src.optimizer.lr_scheduler.NoamScheduler import NoamLR


class PLModel(L.LightningModule):
    """
    Base Class for any Model, includes the optimizer code from config
    """

    def __init__(self, model_cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.verbose = True
        self.cfg = model_cfg
        self.automatic_optimization = model_cfg.automatic_optimization
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        # for e.g. GANs
        self.save_hyperparameters(model_cfg)

    def test_model(self, input_data):
        logger.debug(f"Please implement custom test_model() for {self._get_name()}")

    def configure_optimizers(self) -> Tuple[Union[Optimizer, Sequence[Optimizer]], Union[Scheduler, Sequence[Scheduler]]]:
        """
        Configure the optimizer and learning rate scheduler for the Lightning module.

        Returns:
            A tuple containing one or two elements, depending on whether a learning rate scheduler is used or not.
            The first element is an optimizer or a list of optimizers.
            The second element is a learning rate scheduler or a list of learning rate schedulers.
        """
        logger.info('configure optimizer')
        optim_cfg: DictConfig = self.cfg.Optimizer

        self.learning_rate = optim_cfg.lr
        logger.opt(ansi=True).info(f"learning rate: <yellow>{self.learning_rate}</>")
        self.optimizer_type = optim_cfg.optimizer.lower()
        logger.opt(ansi=True).info(f"optimizer: <yellow>{self.optimizer_type}</>")
        if optim_cfg.maximize:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>maximize</>")
        else:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>minimize</>")

        betas = optim_cfg.get("betas", (0.9, 0.98))
        eps = optim_cfg.get("eps", 1e-9)
        weight_decay = optim_cfg.get("weight_decay", 1e-3)
        momentum = optim_cfg.get("momentum", 0.9)
        # https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), maximize=optim_cfg.maximize, lr=self.learning_rate, betas=betas, eps=eps)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), maximize=optim_cfg.maximize, lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type == "nadam":
            optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
        else:
            logger.exception(f"illegal optimizer: {self.optimizer_type} - suppported optimizer: adam, adamw, sgd, nadam")
            raise ValueError(f"illegal optimizer: {self.optimizer_type} - suppported optimizer: adam, adamw, sgd, nadam")

        scheduler_cfg = optim_cfg.Scheduler

        if scheduler_cfg.type is None or scheduler_cfg.type == "none":
            logger.success('Done: configure optimizer.')
            return [optimizer]

        logger.opt(ansi=True).info(f'scheduler: <yellow>{scheduler_cfg.type}</>')
        if scheduler_cfg.type == 'linear':
            scheduler = (
                {
                    "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                                   start_factor=scheduler_cfg.start_f,
                                                                   end_factor=scheduler_cfg.end_f,
                                                                   total_iters=scheduler_cfg.total_iters,
                                                                   )
                }
            )
        elif scheduler_cfg.type == "plateau":
            scheduler = (
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                            mode=scheduler_cfg.mode,
                                                                            factor=scheduler_cfg.factor,
                                                                            patience=scheduler_cfg.patience,
                                                                            cooldown=scheduler_cfg.cooldown,
                                                                            min_lr=scheduler_cfg.min_lr),
                    "monitor": scheduler_cfg.monitor
                }
            )
        #elif scheduler_cfg.type == "noam":
            #scheduler = (
               #{
                    #"scheduler": NoamLR(optimizer=optimizer, warmup_steps=scheduler_cfg.warmup),
                    #"interval": "step",  # runs per batch rather than per epoch
                    #"frequency": 1,
                #}
            #)
        elif scheduler_cfg.type == "cosine":
            scheduler = (
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                                      T_0=scheduler_cfg.cosine_t0),
                    "interval": "step",  # runs per batch rather than per epoch
                    "frequency": 1,
                }
            )
        else:
            logger.exception(f"Invalid option for scheduler: {scheduler_cfg.type} - supported scheduler: none, noam, cosine")
            raise ValueError(f"Invalid option for scheduler: {scheduler_cfg.type} - supported scheduler: none, noam, cosine")

        logger.success('Done: configure optimizer and lr_scheduler.')
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.train()
        output = self._training_step(batch, batch_idx)
        return output

    def _training_step(self, batch, batch_idx):
        paths, X, y = batch 
        logger.debug(f"please implement training_step() for {self._get_name()}!")

    def validation_step(self, batch, batch_idx):
        self.eval()
        output = self._validation_step(batch, batch_idx)
        return output

    def _validation_step(self, batch, batch_idx):
        paths, X, y = batch
        logger.debug(f"please implement validation_step() for {self._get_name()}!")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.eval()
        output = self._predict_step(batch, batch_idx, dataloader_idx)
        return output

    def _predict_step(self, batch, batch_idx, dataloader_idx):
        paths, X = batch
        logger.debug(f"please implement predict_step() for {self._get_name()}!")

    # -----------------------------------------------------------------------------

    def get_loss(self, y_hat, y):
        loss = self.loss_func(y_hat, y)
        return loss

    def __init_loss_func__(self):
        """Initializes the loss function used for training the model.

        Returns:
            A PyTorch loss function that is either specified in the configuration file or defaults to the mean squared error (MSE) loss function if not found.
        """
        loss_func_name = self.cfg.loss_func.name
        loss_func_args = self.cfg.loss_func.args
        if hasattr(torch.nn, loss_func_name):
            loss_func = getattr(torch.nn, loss_func_name)(**loss_func_args)
            logger.opt(ansi=True).info(f'using loss_func: <y>{loss_func_name}</>')
        elif hasattr(torchmetrics, loss_func_name):
            loss_func = getattr(torchmetrics, loss_func_name)(**loss_func_args)
            logger.opt(ansi=True).info(f'using loss_func: <y>{loss_func_name}</>')
        else:
            logger.error(f'can not find loss_func: {loss_func_name}')
            logger.opt(ansi=True).error('will use <y>mse_loss</>!')
            loss_func = torch.nn.functional.mse_loss

        return loss_func

    # -----------------------------------------------------------------------------

    def get_backbone(self):
        """
        Returns the backbone model for the given configuration.

        Returns:
            The backbone model instance.
        """
        if self.verbose: logger.info('get backbone from config')
        import_string = f"src.model.backbone.{self.cfg.Backbone}"
        module = importlib.import_module(import_string)
        Model = getattr(module, self.cfg.Class_Name)
        model = Model(self.cfg, **self.cfg.Args, verbose=self.verbose)

        return model

    def get_backbone_list(self):
        """
        Returns a list of backbone models based on the configuration specified in `self.cfg.Backbone`.

        Returns:
            List[torch.nn.Module]: A list of PyTorch modules representing the backbones specified in the configuration.
        """
        if self.verbose: logger.info('get backbones from config')
        backbone_list = []
        for backbone_name in self.cfg.Backbone:
            import_string = f"src.model.backbone.{backbone_name}"
            module = importlib.import_module(import_string)
            Model = getattr(module, self.cfg.Class_Name)
            backbone_list.append(Model(self.cfg, **self.cfg.Args, verbose=self.verbose))

        return backbone_list

    # -------------------------------------------------------------------------------

    def save_model(self, model_name, save_dir):
        """
        Save a PyTorch model to a given directory.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            save_dir (str): The directory to save the model to.
        """
        # Create the save directory if it doesn't already exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the model to the directory
        model_path = os.path.join(save_dir, f'{model_name}_epoch={self.current_epoch}_step={self.global_step}.pth')
        torch.save(self.state_dict(), model_path)

        logger.info(f'Model saved to {model_path}')

    @classmethod
    def restore(cls: L.LightningModule, ckpt: str, map_location: Union[torch.device, str, int] = "cpu"):
        """
        Restores a PyTorch Lightning model from a checkpoint file.

        Args:
            cls (LightningModule): The class of the PyTorch Lightning model to restore.
            ckpt (str): The path to the checkpoint file to restore from.
            map_location (Union[torch.device, str, int]): Device to map the restored model to.

        Returns:
            Tuple[LightningModule, DictConfig]: A tuple of the restored model and its configuration.

        Raises:
            RuntimeError: If the checkpoint file does not contain hyperparameters.

        Example:
            # Restore a PLModel from a checkpoint file
            model, config = PLModel.restore(ckpt='path/to/checkpoint.ckpt')
        """
        torch_ckpt = torch.load(ckpt)
        if "hyper_parameters" not in torch_ckpt:
            logger.error("Checkpoint does not contain hyperparameters.")
            raise RuntimeError("Checkpoint does not contain hyperparameters.")

        logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {cls.__name__}\n\tcheckpoint: {ckpt}")
        model = cls.load_from_checkpoint(checkpoint_path=ckpt, map_location=map_location)
        logger.success(f"Successfully loaded checkpoint")

        return model, OmegaConf.create(torch_ckpt["hyper_parameters"])
