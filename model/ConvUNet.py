from typing import Any, Tuple, Union, List
from loguru import logger
from omegaconf import DictConfig

import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
import torchmetrics

from model.convblocks import *

class ConvUNet(L.LightningModule):
    def __init__(self,
                 model_cfg: DictConfig,
                 image_channels: int,
                 output_channels: int,
                 c_factor: int = 6,  # = 2^6 = 64 dim der TimeEmbeddings
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
                 n_blocks: int = 2,
                 n_layers: int = 2,
                 scale_factor: int = 2,
                 kernel_size: int = 3,
                 n_groups: int = 32,
                 verbose: bool = True,
                 *args, **kwargs):
        super().__init__(model_cfg, *args, **kwargs)
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

        # layers:
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
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

        in_channels = n_channels
        
        down = []
        for i in range(n_resolutions - 1):
            out_channels = 2 ** (self.c_factor + self.ch_mults[i])

            for _ in range(self.n_blocks):
                down.append(Conv_Block(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(SkipConnection(in_channels, self.scale_factor, self.n_groups))
        
        self.down = nn.ModuleList(down)

        out_channels = 2 ** (self.c_factor + self.ch_mults[-1])
        self.middle = Conv_Block(in_channels, out_channels, 1, 1, self.kernel_size, n_groups=in_channels) # TODO: check if n_group change is sinnvoll
        in_channels = out_channels

        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = 2 ** (self.c_factor - 1 + self.ch_mults[i])

            for _ in range(self.n_blocks):
                up.append(Conv_Block(in_channels, out_channels, self.n_layers, 1, self.kernel_size, self.n_groups))
                in_channels = out_channels

            if i > 0:
                up.append(CombineConnection(in_channels, self.scale_factor, self.n_groups))

        self.up = nn.ModuleList(up) 

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1)

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

    def training_step(self, batch, batch_idx):
        self.train()
        output = self._training_step(batch, batch_idx)
        return output

    def _training_step(self, batch, batch_idx):
        x, y = batch 
        logger.debug(f"please implement training_step() for {self._get_name()}!")
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)

        return {'loss': loss, 'log': {'train_loss': loss}}
                
    def validation_step(self, batch, batch_idx):
        self.eval()
        output = self._validation_step(batch, batch_idx)
        return output

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        logger.debug(f"please implement validation_step() for {self._get_name()}!")
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)

        return {'val_loss': loss}

    def get_loss(self, y_hat, y):
        loss = self.__init_loss_func__(y_hat, y)
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
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the Lightning module.

        Returns:
            A tuple containing one or two elements, depending on whether a learning rate scheduler is used or not.
            The first element is an optimizer or a list of optimizers.
            The second element is a learning rate scheduler or a list of learning rate schedulers.
        """
        return torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=1e-8)
    
