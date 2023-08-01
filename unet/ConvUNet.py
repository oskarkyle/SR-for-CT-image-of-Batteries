from typing import Any, Tuple, Union, List, Sequence
from loguru import logger
from omegaconf import DictConfig
import lightning as L
from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer

import os
import torch
from torch import nn
#from torchsummary import summary
#from src.model.base.BaseModel import PLModel

from unet.blocks import Swish, ConvBlock, SkipConnection, CombineConnection
#from model.activations.activation import Swish

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
                 #model_cfg: DictConfig,
                 image_channels: int = 1,
                 output_channels: int = 1,
                 c_factor: int = 6,  # = 2^6 = 64 dim der TimeEmbeddings
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
                 n_blocks: int = 2,
                 n_layers: int = 2,
                 scale_factor: int = 2,
                 kernel_size: int = 3,
                 n_groups: int = 32,
                 verbose: bool = True,
                 lr: float = 1e-3,
                 optimizer_type: str = "adam",
                 maximize: bool = False
                 ):
        super(ConvUNet, self).__init__()
        self.verbose = verbose
        # hyperparameter:
        self.c_factor = c_factor
        if isinstance(ch_mults, int):
            self.ch_mults = [m+1 for m in range(ch_mults)]
        else:
            self.ch_mults = ch_mults
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.learning_rate = lr
        self.optimizer_type = optimizer_type
        self.maximize = maximize
        self.image_channels = image_channels
        self.output_channels = output_channels


        if verbose:
            logger.info("Build ConvUNet")
            logger.opt(ansi=True).info(f"Image Channels: <yellow>{image_channels}</>")
            logger.opt(ansi=True).info(f"Channel Factor: <yellow>{self.c_factor}</>")
            logger.opt(ansi=True).info(f"Channel Multiplyer: <yellow>{self.ch_mults}</>")
            logger.opt(ansi=True).info(f"Number of Blocks: <yellow>{self.n_blocks}</>")
            logger.opt(ansi=True).info(f"Number of Layers: <yellow>{self.n_layers}</>")
            logger.opt(ansi=True).info(f"Number of Groups in GroupNorm: <yellow>{self.n_groups}</>")

        n_channels = 2 ** self.c_factor
        n_resolutions = len(self.ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)) # Change

        in_channels = n_channels

        down = []
        for i in range(n_resolutions - 1):
            out_channels = 2 ** (self.c_factor + self.ch_mults[i])

            for _ in range(self.n_blocks):
                down.append(ConvBlock(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(SkipConnection(in_channels, self.scale_factor, self.n_groups))

        self.down = nn.ModuleList(down)

        out_channels = 2 ** (self.c_factor + self.ch_mults[-1])
        self.middle = ConvBlock(in_channels, out_channels, 1, 1, self.kernel_size, n_groups=in_channels) # TODO: check if n_group change is sinnvoll
        in_channels = out_channels

        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = 2 ** (self.c_factor - 1 + self.ch_mults[i])

            for _ in range(self.n_blocks):
                up.append(ConvBlock(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i > 0:
                up.append(CombineConnection(in_channels, self.scale_factor, self.n_groups))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)) # Change

        self.sig = nn.Sigmoid()

        logger.success('Done: Create ConvUNet.')

    def forward(self, x: torch.Tensor):
        x = self.image_proj(x)

        h = []

        for m in self.down:
            if isinstance(m, SkipConnection):
                x, x_e = m(x)
                h.append(x_e)
            else:
                x = m(x)


        x = self.middle(x)

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
        self.verbose: logger.success(f"Done: Test passed")

    def configure_optimizers(self) -> Tuple[Union[Optimizer, Sequence[Optimizer]], Union[Scheduler, Sequence[Scheduler]]]:
        """
        Configure the optimizer and learning rate scheduler for the Lightning module.

        Returns:
            A tuple containing one or two elements, depending on whether a learning rate scheduler is used or not.
            The first element is an optimizer or a list of optimizers.
            The second element is a learning rate scheduler or a list of learning rate schedulers.
        """
        logger.info('configure optimizer')
        #optim_cfg: DictConfig = self.cfg.Optimizer

        #self.learning_rate = lr#optim_cfg.lr
        logger.opt(ansi=True).info(f"learning rate: <yellow>{self.learning_rate}</>")
        #self.optimizer_type = optim_cfg.optimizer.lower()
        logger.opt(ansi=True).info(f"optimizer: <yellow>{self.optimizer_type}</>")
        '''if optim_cfg.maximize:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>maximize</>")
        else:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>minimize</>")'''
        if self.maximize:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>maximize</>")
        else:
            logger.opt(ansi=True).info(f"optimizer strategy: <yellow>minimize</>")

        betas = (0.9, 0.98)#optim_cfg.get("betas", (0.9, 0.98)
        eps = 1e-9#optim_cfg.get("eps", 1e-9)
        weight_decay = 1e-3#optim_cfg.get("weight_decay", 1e-3)
        momentum = 0.9#optim_cfg.get("momentum", 0.9)
        # https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), maximize=self.maximize, lr=self.learning_rate, betas=betas, eps=eps)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), maximize=self.maximize, lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type == "nadam":
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
        else:
            logger.exception(f"illegal optimizer: {self.optimizer_type} - suppported optimizer: adam, adamw, sgd, nadam")
            raise ValueError(f"illegal optimizer: {self.optimizer_type} - suppported optimizer: adam, adamw, sgd, nadam")

        logger.success('Done: configure optimizer.')
        return [optimizer]


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.get_loss(outputs, labels)

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.get_loss(outputs, labels)

        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        predictions = self.forward(x)
        return predictions

    def get_loss(self, y_hat, y):
        loss = torch.nn.functional.mse_loss(y_hat, y)#self.loss_func(y_hat, y)
        return loss
        
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
        """
        torch_ckpt = torch.load(ckpt)
        if "hyper_parameters" not in torch_ckpt:
            logger.error("Checkpoint does not contain hyperparameters.")
            raise RuntimeError("Checkpoint does not contain hyperparameters.")
            """

        logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {cls._get_name}\n\tcheckpoint: {ckpt}")
        model = cls.load_from_checkpoint(checkpoint_path=ckpt, map_location=map_location)
        logger.success(f"Successfully loaded checkpoint")
        

        return model#, OmegaConf.create(torch_ckpt["hyper_parameters"])

if __name__ == "__main__":
    model = ConvUNet()
    #model.test_model(torch.rand(8, 1, 512, 512))
    #summary(model, (1, 256, 256), batch_size=8)

    '''
    x = torch.rand(8, 1, 512, 512)
    train_dataloader = torch.utils.data.DataLoader(x, batch_size=1)
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader)
    '''

    print(isinstance(model, L.LightningModule))