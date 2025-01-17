from typing import Any, Tuple, Union, List, Sequence
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import lightning as L
from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer
import torchmetrics


import os
import torch
from torch import nn

from model.unet.blocks import Swish, ConvBlock, SkipConnection, CombineConnection
from model.unet.lr_scheduler import NoamLR

class ConvUNet(L.LightningModule):
    """A simple Conv U-Net.

    Args:
        model_cfg (DictConfig): The configuration dictionary for the model.        
        image_channels (int): The number of channels in the input image.      
        output_channels (int): The number of channels in the output image.       
        c_factor (int, optional): The channel factor. Defaults to 6.       
        ch_mults (Union[Tuple[int, ...], List[int]], optional): The channel multipliers. Defaults to (1, 2, 3, 4).      
        n_blocks (int, optional): The number of blocks. Defaults to 2.      
        n_layers (int, optional): The number of layers. Defaults to 2.        
        scale_factor (int, optional): The scale factor. Defaults to 2.        
        kernel_size (int, optional): The kernel size. Defaults to 3.      
        n_groups (int, optional): The number of groups in GroupNorm. Defaults to 32.    
        verbose (bool, optional): Whether to print verbose logs. Defaults to True.

    Attributes:
        image_proj (nn.Conv2d): The image projection layer.       
        down (nn.ModuleList): The down-sampling layers.        
        middle (ConvBlock): The middle layer.       
        up (nn.ModuleList): The up-sampling layers.       
        norm (nn.GroupNorm): The group normalization layer.      
        act (Swish): The activation function.      
        final (nn.Conv2d): The final convolutional layer.   
        sig (nn.Sigmoid): The sigmoid activation function.

    """
    def __init__(self,
                 model_cfg: DictConfig,
                 image_channels: int = 1,
                 output_channels: int = 1,
                 c_factor: int=6,  # = 2^6 = 64 dim der TimeEmbeddings
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
                 n_blocks: int = 2,
                 n_layers: int = 2,
                 scale_factor: int = 2,
                 kernel_size: int = 3,
                 n_groups: int = 32,
                 verbose: bool = True
                 ):
        """
        To construct the ConvUNet model with the given hyperparameters.
        The Architecture of ConvUNet is built from basic blocks including ConvBlock, SkipConnection and CombineConnection.
        """
        super(ConvUNet, self).__init__()
        self.verbose = verbose
        self.cfg = model_cfg
        # hyperparameter:
        # Exponential factor over 2, for image projection
        self.c_factor = c_factor
        # Exponential factor to be added onto c_factor
        if isinstance(ch_mults, int):
            self.ch_mults = [m+1 for m in range(ch_mults)]
        else:
            self.ch_mults = ch_mults
        # ConvBlock manipulation in each combine or skip connection     
        self.n_blocks = n_blocks
        # Conv2d layers in ConvBlock, 2 for direct output from input
        self.n_layers = n_layers
        
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.image_channels = image_channels
        self.output_channels = output_channels

        self.loss_func = self.__init_loss_func__()
        self.save_hyperparameters(model_cfg)

        if verbose:
            logger.info("Build ConvUNet")
            logger.opt(ansi=True).info(f"Image Channels: <yellow>{image_channels}</>")
            logger.opt(ansi=True).info(f"Channel Factor: <yellow>{self.c_factor}</>")
            logger.opt(ansi=True).info(f"Channel Multiplyer: <yellow>{self.ch_mults}</>")
            logger.opt(ansi=True).info(f"Number of Blocks: <yellow>{self.n_blocks}</>")
            logger.opt(ansi=True).info(f"Number of Layers: <yellow>{self.n_layers}</>")
            logger.opt(ansi=True).info(f"Number of Groups in GroupNorm: <yellow>{self.n_groups}</>")

        # Exponential factor over 2, for image projection 
        n_channels = 2 ** self.c_factor
        # Times for multiplication
        n_resolutions = len(self.ch_mults)

        # Projection block
        # The assigned value is also the group numbers since it is dividable by all the rest blocks 
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)) # Change
        # Starting the input channels from the projection 
        in_channels = n_channels
        # Downsampling with skip and residual
        down = []
        for i in range(n_resolutions - 1):
            out_channels = 2 ** (self.c_factor + self.ch_mults[i])

            for _ in range(self.n_blocks):
                down.append(ConvBlock(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(SkipConnection(in_channels, self.scale_factor, self.n_groups))

        self.down = nn.ModuleList(down)

        # Apply an extra ConvBloak to reach maximum output channels, reset in_channels with maximum for later upsampling
        # Size upsampling corresponds to channel downgrade and vice versa.
        out_channels = 2 ** (self.c_factor + self.ch_mults[-1])
        self.middle = ConvBlock(in_channels, out_channels, 1, 1, self.kernel_size, n_groups=in_channels) # TODO: check if n_group change is sinnvoll
        in_channels = out_channels

        #  Upsampling with addition from residual
        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = 2 ** (self.c_factor - 1 + self.ch_mults[i])

            for _ in range(self.n_blocks):
                up.append(ConvBlock(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i > 0:
                up.append(CombineConnection(in_channels, self.scale_factor, self.n_groups))

        self.up = nn.ModuleList(up)

        # Prepare the blocks for final output 
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)) # Change

        self.sig = nn.Sigmoid()

        logger.success('Done: Create ConvUNet.')

    def forward(self, x: torch.Tensor):
        """
        The forward method of the ConvUNet class defines the forward pass of the convolutional U-Net model. It takes an input tensor x and passes it through the layers of the model to produce an output tensor.

        The method starts by passing the input tensor through the image projection layer. It then performs downsampling using skip and residual connections, saving the output tensor from each skip connection in a list h for later combination.

        After the downsampling layers, the method passes the output tensor through the middle layer of the model. It then performs upsampling using the saved skip connection tensors and residual connections, combining them with the output tensor from each upsampling layer.

        Finally, the method passes the output tensor through the final layer of the model, applies an activation function, and returns the resulting tensor.
        """

        # Starting the input channels from the projection 
        x = self.image_proj(x)
        # Downsampling with skip and residual
        h = []

        for m in self.down:
            if isinstance(m, SkipConnection):
                x, x_e = m(x)
                h.append(x_e)
            else:
                x = m(x)

        # Middle layer
        x = self.middle(x)
        # Upsampling with addition from residual
        for m in self.up:
            if isinstance(m, CombineConnection):
                s = h.pop()
                x = m(x, s)
            else:
                x = m(x)

        return self.sig(self.final(self.act(self.norm(x))))

    def test_model(self, input_data):
        x = input_data

        assert isinstance(x, torch.Tensor), logger.exception(f"test input data does not fit for this model.")
        assert len(x.shape) == 4, logger.exception(f"test input data does not fit for this model.")

        if self.verbose: logger.info(f"Test model forward pass")
        y = self(x)
        logger.success(f"Done: Test passed")

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
            optimizer = torch.optim.Adam(self.parameters(), maximize=optim_cfg.maximize, lr=self.learning_rate, betas=betas, eps=eps)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), maximize=optim_cfg.maximize, lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type == "nadam":
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
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
        elif scheduler_cfg.type == "noam":
            scheduler = (
                {
                    "scheduler": NoamLR(optimizer=optimizer, warmup_steps=scheduler_cfg.warmup),
                    "interval": "step",  # runs per batch rather than per epoch
                    "frequency": 1,
                }
            )
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
        """
        Training step for the lightning module.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for the current training step.
        """
        self.train()
        loss = self._training_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True) 
        return loss

    def _training_step(self, batch, batch_idx):

        """
        x: transformed image tensors as input
        y: original image tensors as label
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat,y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the lightning module.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for the current validation step.
        """

        self.eval()
        loss = self._validation_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        """
        x: transformed image tensors as input
        y: original image tensors as label
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat,y) 
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Prediction for the lightning module.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            torch.Tensor: The prediction of current batch.
        """
        output = self._predict_step(batch, batch_idx, dataloader_idx)
        return output

    def _predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        """
        x: transformed image tensors as input
        y: original image tensors as label
        """
        y_hat = self.forward(x)
        return y_hat

    # -----------------------------------------------------------------------------
    def get_loss(self, y_hat, y):
        """
        To get the loss from the loss function

        Returns:
            torch.Tensor: The loss for the current step.
        """

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
    def restore(cls: L.LightningModule, ckpt: str, map_location: Union[torch.device, str, int] = "mps"):
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


        torch_ckpt = torch.load(ckpt, map_location=map_location)
        if "hyper_parameters" not in torch_ckpt:
            logger.error("Checkpoint does not contain hyperparameters.")
            raise RuntimeError("Checkpoint does not contain hyperparameters.")


        logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {cls.__name__}\n\tcheckpoint: {ckpt}")
        model = cls.load_from_checkpoint(checkpoint_path=ckpt, map_location=map_location)
        logger.success(f"Successfully loaded checkpoint")

        return model, OmegaConf.create(torch_ckpt["hyper_parameters"])

