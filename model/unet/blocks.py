import torch
from torch import nn

#from model.activations.activation import Swish

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

"""
        Conv block
"""
class ConvBlock(nn.Module):
    """
    A block of convolutional layers followed by max pooling, group normalization and activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_layers (int): Number of convolutional layers.
        scale_factor (int): Factor by which to downsample the input.
        kernel_size (int): Size of the convolutional kernel.
        n_groups (int): Number of groups to use for group normalization.

    Returns:
        torch.Tensor: The output tensor.
"""
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 2, scale_factor: int = 2, kernel_size: int = 3, n_groups: int = 32):
        super().__init__()

        assert n_layers > 0, "n_layer must be greater than 0"

        c_list = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), padding=(1, 1))]
        for _ in range(n_layers-1):
            c_list.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), padding=(1, 1)))

        self.conv_layers = nn.ModuleList(c_list)
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

        if n_groups is not None:
            self.norm = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm = nn.Identity()

        self.act = Swish()

    def forward(self, x: torch.Tensor):
        for conv in self.conv_layers:
            x = conv(x)
        x = self.pool(self.act(self.norm(x)))
        return x

"""
        DeConv block
"""
class DeConvBlock(nn.Module):
    """
    A block of transposed convolutional layers followed by upsampling, group normalization and activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_layers (int): Number of transposed convolutional layers.
        scale_factor (int): Factor by which to upsample the input.
        kernel_size (int): Size of the transposed convolutional kernel.
        n_groups (int): Number of groups to use for group normalization.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 2, scale_factor: int = 2,
                 kernel_size: int = 3, n_groups: int = 32):
        super().__init__()

        assert n_layers > 0, "n_layer must be greater than 0"

        c_list = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                            padding=(1, 1))]
        for _ in range(n_layers-1):
            c_list.append(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                          padding=(1, 1)))

        self.conv_layers = nn.ModuleList(c_list)

        self.scale = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        if n_groups is not None:
            self.norm = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm = nn.Identity()

        self.act = Swish()

    def forward(self, x: torch.Tensor):
        x = self.scale(x)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.act(self.norm(x))
        return x

"""
        Skip and Combine Connection Blocks
"""

class SkipConnection(nn.Module):
    """
    A block that performs a skip connection by max pooling the input tensor and applying group normalization and activation
    function to it.

    Args:
        channels (int): Number of input channels.
        scale_factor (int): Factor by which to downsample the input.
        n_groups (int): Number of groups to use for group normalization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor and the intermediate tensor.
    """
    def __init__(self, channels, scale_factor: int = 2, n_groups: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        if n_groups is not None:
            self.norm = nn.GroupNorm(n_groups, channels)
        else:
            self.norm = nn.Identity()
        self.act = Swish()

    def forward(self, x: torch.Tensor):
        x_e = self.act(self.norm(x))
        x = self.pool(x_e)
        return x, x_e

class CombineConnection(nn.Module):
    """
    A block that combines the skip connection and residual block by performing a residual connection between the input tensor and
    the output of the residual block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Factor by which to upsample the input.
        n_groups (int): Number of groups to use for group normalization.
        dropout (float): Dropout probability.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, channels, scale_factor: int = 2, n_groups: int = 32):
        super().__init__()
        self.scale = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        if n_groups is not None:
            self.norm = nn.GroupNorm(n_groups, channels)
        else:
            self.norm = nn.Identity()
        self.act = Swish()

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        x = self.act(self.norm(self.scale(x)))
        x = torch.add(x, x_e)
        return x

"""
        Residual block
"""
class ResidualBlock(nn.Module):
    """
    A residual block that consists of two convolutional layers with group normalization and activation functions, and a
    residual connection between the input tensor and the output of the second convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_groups (int): Number of groups to use for group normalization.
        dropout (float): Dropout probability.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.shortcut(x)

"""
        Attention block
"""
class CustomAttentionBlock(nn.Module):
    """
    A custom attention block that performs multi-head self-attention on the input tensor.

    Args:
        n_channels (int): Number of input channels.
        n_heads (int): Number of attention heads.
        d_k (int): Dimensionality of the key and query vectors.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        super().__init__()

        if d_k is None:
            d_k = n_channels #// n_heads <- funst

        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

        #self.attn = nn.MultiheadAttention(d_k, n_heads)

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1) # [batch_size, n_channels, height, width] -> [batch_size, height * width, n_channels]
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k) # res.view(batch_size, -1, self.n_heads * self.d_k)

        res = self.output(res)
        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res

class MultiheadAttentionBlock(nn.Module):
    """
    A multi-head attention block that applies self-attention to the input tensor.

    Args:
        n_channels (int): Number of input channels.
        n_heads (int): Number of attention heads.
        d_k (int): Dimensionality of the key and query vectors.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        super().__init__()

        if d_k is None:
            d_k = n_channels #// n_heads <- funst nicht

        self.attn = nn.MultiheadAttention(n_channels, n_heads, kdim=d_k, vdim=d_k)
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape

        # reshape to [seq_len, batch_size, embed_dim] when batch_first=False (default)
        x = x.permute(2, 3, 0, 1)  # [batch_size, n_channels, height, width] -> [height, width, batch_size, n_channels]
        x = x.view(height * width, batch_size, n_channels)

        # apply attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x) # Self-Attention
        x = x + residual
        x = self.norm2(x)

        # reshape back to [batch_size, n_channels, height, width]
        x = x.view(height, width, batch_size, n_channels)
        x = x.permute(2, 3, 0, 1)  # [batch_size, n_channels, height, width]

        return x


"""
        Combinations
"""
class ResAttnBlock(nn.Module):
    """
    A residual attention block that applies multi-head self-attention to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        has_attn (bool): Whether to apply attention or not.
        n_heads (int): Number of attention heads.
        n_groups (int): Number of groups for group normalization.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, n_heads: int=1, n_groups: int = 32):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, n_groups)

        if has_attn:
            self.attn = MultiheadAttentionBlock(out_channels, n_heads=n_heads)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)

        return x

class BottleneckResAttnBlock(nn.Module):
    """
    A bottleneck residual attention block that applies multi-head self-attention to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        n_channels (int): Number of intermediate channels.
        n_heads (int): Number of attention heads.
        n_groups (int): Number of groups for group normalization.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, in_channels: int, n_channels: int, n_heads: int = 1, n_groups: int = 32):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, n_channels, n_groups)
        self.attn = MultiheadAttentionBlock(n_channels, n_heads)
        self.res2 = ResidualBlock(n_channels, in_channels, n_groups)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x

"""
        Downsample
"""
class Downsample(nn.Module):
    """
    Downsamples the input tensor by a factor of 2 using a 3x3 convolution with stride 2.

    Args:
        n_channels (int): Number of input channels.
        scale_factor (int): Factor by which to downsample the input tensor. Default: 2.

    Returns:
        torch.Tensor: The downsampled tensor.
    """
    def __init__(self, n_channels, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (scale_factor, scale_factor), (1, 1))

    def forward(self, x: torch.Tensor, ):
        return self.conv(x)

"""
        Upsample    
"""
class Upsample(nn.Module):
    """
    Upsamples the input tensor by a factor of 2 using a 4x4 transposed convolution with stride 2.

    Args:
        n_channels (int): Number of input channels.
        scale_factor (int): Factor by which to upsample the input tensor. Default: 2.

    Returns:
        torch.Tensor: The upsampled tensor.
    """
    def __init__(self, n_channels, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (scale_factor, scale_factor), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)