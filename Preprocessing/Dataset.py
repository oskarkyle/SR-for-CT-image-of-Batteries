import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning as L

class MyDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.transform = transforms.ToTensor
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.file_list[index]))
        img = self.transform(img)

        return img
    
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        # To create train set and val set
        self.train_dataset = MyDataset(os.path.join(self.data_path, 'bin_2'))
        self.val_dataset = MyDataset(os.path.join(self.data_path, 'original'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers=4, pin_memory=True)    
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=False, num_workers=4, pin_memory=True)