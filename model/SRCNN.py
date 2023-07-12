from torch import nn
from omegaconf import DictConfig, OmegaConf

class SRCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SRCNN, self).__init__()
        # Define the layers
        """
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        """

        self.num_channels = cfg.model.num_channels
        self.num_features = cfg.model.num_features
        self.kernel_size = cfg.model.kernel_size

        self.conv1 = nn.Conv2d(self.num_channels, self.num_features, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.conv2 = nn.Conv2d(self.num_features, self.num_features // 2, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(self.num_features // 2, self.num_channels, kernel_size=5, padding=5 // 2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
