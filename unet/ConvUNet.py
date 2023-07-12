from typing import Tuple, Union, List
from loguru import logger
from omegaconf import DictConfig
import pytorch_lightning as L

import torch
from torch import nn

#from src.model.base.BaseModel import PLModel

from blocks import Swish, ConvBlock, SkipConnection, CombineConnection
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
        super().__init__(*args, **kwargs)
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
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

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
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

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

model = ConvUNet(image_channels=1, output_channels=1)
model.test_model(torch.rand(8, 1, 256, 256))