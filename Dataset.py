import os
from loguru import logger
from typing import Iterable, Optional, Sequence, Union
from typing import Tuple, Optional, List, Union
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from torchvision import transforms
from pathlib import Path
from natsort import natsorted
from preprocess.Preprocessor import Preprocessor
from transform.Transforms import Transforms

from omegaconf import DictConfig, ListConfig,OmegaConf
import matplotlib.pyplot as plt
# import lightning as L


class MyDataset(Dataset):
    def __init__(self, binning_data_dir, image_label_dir, transform=None):
        self.binning_data_dir = binning_data_dir
        self.image_label_dir = image_label_dir

        self.transform = transform

        self.filesnames = natsorted(os.listdir(binning_data_dir))
    '''
    def load_files_from_dir(directory):
        file_list = []

        # Iterate over all files in the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)

            # Check if the path points to a file
            if os.path.isfile(file_path):
                file_list.append(file_path)

        return natsorted(file_list)
    '''
    
    def __len__(self):
        return len(self.filesnames)

    def __getitem__(self, idx):
        binning_names = self.filesnames[idx]
        label_names = binning_names

        print(os.path.join(self.binning_data_dir, binning_names))
        print(os.path.join(self.image_label_dir, label_names))

        binning = Image.open(os.path.join(self.binning_data_dir, binning_names))
        image_label = Image.open(os.path.join(self.image_label_dir, label_names))
        '''
        binning_array = Image.fromarray(binning)
        image_label_array = Image.fromarray(image_label)

        plt.subplot(131)
        plt.imshow(binning_array, cmap='gray')
        plt.title('data')

        plt.subplot(132)
        plt.imshow(image_label_array)
        plt.title('label')
        '''
        if self.transform:
            binning = self.transform(binning)
            image_label = self.transform(image_label)

        binning = transforms.ToTensor()(binning)
        image_label = transforms.ToTensor()(image_label)
        return binning, image_label

class MyDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True, num_workers=0, collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def collate_fn(self, batch):
        binning, image_label = zip(*batch)
        return torch.stack(binning), torch.stack(image_label)




def tensorToimg(image):
    image = transforms.ToPILImage()(image)
    image.show()



if __name__ == "__main__":
    input_path = "/Users/haoruilong/Dataset_for_Battery/PTY_raw/data"
    label_path = "/Users/haoruilong/Dataset_for_Battery/PTY_raw/label"

   
    dataset = MyDataset(input_path, label_path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    index = 300

    #binning, image_label = dataset.__getitem__(index)
    '''
    print(binning.shape)
    print(image_label.shape)
    # binning = transforms.ToPILImage()(binning)
    # image_label = transforms.ToPILImage()(image_label)
    tensorToimg(binning)
    tensorToimg(image_label)
    # binning.save(f"./images/binning_{index}.jpeg", "JPEG")
    # image_label.save(f"./images/image_label_{index}.jpeg", "JPEG")
    
    dataloader = MyDataLoader(dataset, batch_size=5, shuffle=True)
    for binning, image_label in dataloader:
        print(binning.shape)
        print(image_label.shape)

        for i in range(binning.size(0)):
            binning_tensor = binning[i]
            bin = transforms.ToPILImage()(binning_tensor)
            bin.save(f'./images/binning_{i}.png', 'PNG')
    
        for j in range(image_label.size(0)):
            image_label_tensor = image_label[j]
            label = transforms.ToPILImage()(image_label_tensor)
            label.save(f'./images/image_label_{j}.png', 'PNG')
        break
    '''

    for binning, image_label in dataloader:
        print(binning.shape)
        print(image_label.shape)
