import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, List, Union
import tifffile
from natsort import natsorted
import torchvision.transforms as transforms
import torch
import time
from loguru import logger
from transform.Transforms import Transforms
from preprocess.Preprocessor import Preprocessor
from image_utils import tiling
from pathlib import Path

def disassemble_dataloader(dataloader):
    input = []
    label = []

    for data in dataloader:
        inputs, labels = data
        for img in inputs:
            img = transforms.ToPILImage()(img)
            input.append(img)

        for img in labels:
            img = transforms.ToPILImage()(img)
            label.append(img)

    return input, label

# To plot the images from the prediction
def plot(predictions: list, pred_dataloader: DataLoader):
    pred = []
    for img_batch in predictions:
        for img in img_batch:
            img = img.squeeze(0)
            img_array = img.numpy()

            pred.append(img_array)

    input, label = disassemble_dataloader(pred_dataloader)
    
    vmin = 0
    vmax = 1
    
    if len(pred) == len(input):
        for i in range(len(pred)):
            
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

            ax[0].imshow(input[i], cmap='gray')
            title = ax[0].set_title('Input')
            title.set_fontsize(20)
            ax[1].imshow(pred[i], cmap='gray', vmin = vmin, vmax = vmax)
            title = ax[1].set_title('Prediction')
            title.set_fontsize(20)
            ax[2].imshow(label[i], cmap='gray')
            title = ax[2].set_title('Label')
            title.set_fontsize(20)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'./results/predictions_{i}.png')

def check_dataset(dataset, cfg: DictConfig):
        length = dataset.__len__()
        print(length)
        if length > 0:
            print('Dataset is OK')
        print('Checking dataset...')
        new_length = cfg.length
        for i in range(new_length):
            preprocess, transform, label = dataset.__getitem__(i)

            # print(i, input.shape, label.shape)
            preprocess = preprocess.squeeze(0).numpy()
            transform = transform.squeeze(0).numpy()
            label = label.squeeze(0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 5))
            axes[0].imshow(preprocess, cmap='gray')
            title = axes[0].set_title('Preprocess')
            title.set_fontsize(20)
            axes[1].imshow(transform, cmap='gray')
            title = axes[1].set_title('Transform')
            title.set_fontsize(20)
            axes[2].imshow(label, cmap='gray')
            title = axes[2].set_title('Normal')
            title.set_fontsize(20)

            #plt.savefig(f'./results/check_{i}.png')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            time.sleep(0.5)           
        print('Dataset is checked!')


class Pred_dataset(Dataset):
    def __init__(self,
                 task,
                 mode,
                 tile_size, 
                 tile_grid,
                 dataset_dir: Union[List[str], str],
                 data_root: str = "",
                 transforms_cfg: DictConfig = None,
                 preprocess_cfg: DictConfig = None
                 ):
        """Initializes a new instance of the `BaseDataset` class.

        Args:
            task (str): The name of the task.
            mode (str): The mode in which the dataset is being used (e.g., 'train', 'val', 'test').
            dataset_dir (Union[List[str], str]): The path or list of paths to the dataset directory(ies).
            data_root (str, optional): The root directory of the dataset. Defaults to "".
            transforms_cfg (DictConfig, optional): The configuration for the data transforms. Defaults to None.
            preprocess_cfg (DictConfig, optional): The configuration for the data pre-processing. Defaults to None.

        Raises:
            ValueError: If no dataset directory is given.

        """
        if isinstance(dataset_dir, str):
            logger.opt(ansi=True).info(f'<yellow>{dataset_dir}</>')
            dataset_dir = [dataset_dir]
        elif isinstance(dataset_dir, ListConfig) or isinstance(dataset_dir, List):
            for dir in dataset_dir:
                logger.opt(ansi=True).info(f'<yellow>{dir}</>')
        else:
            logger.exception(f"no dataset directory given!")
            raise ValueError(f"no dataset directory given!")

        self.task = task
        self.X_mode = mode
        self.data_root = data_root
        self.tile_size = tile_size
        self.file_list = self.load_file_paths_from_dir(data_root, dataset_dir)
        self.tile_grid = tile_grid
        #self.dataset = self.prepare_data_list(self.file_list)
        self.tiles_count_in_each_tiff = []
        self.tiling_grid_in_each_tiff = []
        self.page_count_in_each_tiff = []
        self.tiff_intersection_border = [0]
        #self.check_dataset_constrains()
        self.transform_cfg = None # transforms_cfg
        self.transform = self.get_transforms(transforms_cfg)
        self.preprocess_cfg = preprocess_cfg
        self.preprocess = self.get_preprocessor(preprocess_cfg)
        for file in self.file_list:
            page_count_in_current_tiff = self.get_pages_count_in_tile(file)
            tiles_count_in_current_tiff = self.get_tiles_count_in_each_tiff(file)
            tiling_grid_in_current_tiff = self.get_tiling_grid_in_each_tiff(file)
            self.tiles_count_in_each_tiff.append(tiles_count_in_current_tiff)
            self.page_count_in_each_tiff.append(page_count_in_current_tiff)
            self.tiling_grid_in_each_tiff.append(tiling_grid_in_current_tiff)
            self.tiff_intersection_border.append(self.tiff_intersection_border[-1] + tiles_count_in_current_tiff)

    def get_pages_count_in_tile(self, tiff_file):
        with tifffile.TiffFile(tiff_file) as handle:
            num_pages = len(handle.pages)
        return num_pages

    def get_tiles_count_in_each_tiff(self, tiff_file):
        tile_size = self.tile_size
        pages_count_in_tiff = self.get_pages_count_in_tile(tiff_file)

        with tifffile.TiffFile(tiff_file) as handle:
            current_page = handle.pages[0]
                
        tiling_grid_info = [self.tile_grid, self.tile_grid]#tiling.get_tiling_grid(current_page, tile_size)
        tiles_count_in_current_page = tiling_grid_info[0] * tiling_grid_info[1]
        tiles_count_in_each_tiff = pages_count_in_tiff * tiles_count_in_current_page
        
        return tiles_count_in_each_tiff
    
    def get_tiling_grid_in_each_tiff(self, tiff_file):
        tile_size = self.tile_size
        
        with tifffile.TiffFile(tiff_file) as handle:
            current_page = handle.pages[0]
                
        tiling_grid_info = [self.tile_grid, self.tile_grid]#tiling.get_tiling_grid(current_page, tile_size)  
        return tiling_grid_info   

    def get_item_position(self, idx):
        left = 0
        right = len(self.file_list) - 1
        tifffile_index = -1

        while left <= right:
            mid = (left + right) // 2

            if self.tiff_intersection_border[mid] <= idx:
                tifffile_index = mid
                left = mid + 1
            else:
                right = mid - 1
        tile_offset_in_current_tifffile = idx - self.tiff_intersection_border[tifffile_index]
        tiles_number_in_page = self.tiling_grid_in_each_tiff[tifffile_index][0] * self.tiling_grid_in_each_tiff[tifffile_index][1]
        page_index = tile_offset_in_current_tifffile // (tiles_number_in_page)
        
        # Postion in grid to locate a tile in current page
        tiling_grid_info = self.tiling_grid_in_each_tiff[tifffile_index]
        sequence_number_of_tile_in_page = tile_offset_in_current_tifffile - tiling_grid_info[0] * tiling_grid_info[1] * page_index

        #logger.info(f"tiff_file index: {tifffile_index}, sequence number of tile in page: {sequence_number_of_tile_in_page}, page index: {page_index}, tile_grid: {tiling_grid_info}")
        
        return tifffile_index,page_index,sequence_number_of_tile_in_page,tiling_grid_info


    
    def get_tile_from_index(self, index):
        tifffile_index,page_index,sequence_number_of_tile_in_page,tiling_grid_info = self.get_item_position(index)

        tiff_file = self.file_list[tifffile_index]
        

        with tifffile.TiffFile(tiff_file) as handle:
            current_page = handle.pages[page_index]
            page_array = current_page.asarray()
            page_tensor = torch.from_numpy(page_array)
            tile_size = self.tile_size
            tile = tiling.get_tile_by_sequence_number(page_tensor, sequence_number_of_tile_in_page, tiling_grid_info)
            if tile.shape != (tile_size, tile_size):
                tile = tiling.pad(tile, tile_size)

            return tile


    @staticmethod
    def get_transforms(transforms_cfg: DictConfig) -> Optional[Transforms]:
        """
        Returns a Transform object based on the provided configuration.

        Args:
            transforms_cfg (DictConfig): The configuration for the transforms.

        Returns:
            Transforms: The Transform object.

        Notes:
            This method creates a Transforms object based on the provided configuration.
            If no configuration is provided (i.e., if transforms_cfg is None), None is returned.
        """
        if transforms_cfg is None: return None
        cfg = transforms_cfg.copy()
        return Transforms(cfg)

    @staticmethod
    def get_preprocessor(preprocess_cfg: DictConfig) -> Optional[Preprocessor]:
        """
        Returns a Preprocessor object based on the provided preprocessing configuration.

        Args:
            preprocess_cfg (DictConfig): A configuration object that specifies the details of the preprocessing pipeline.

        Returns:
            Preprocessor: A Preprocessor object that can be used to preprocess the data.

        Notes:
            This method creates a Preprocessor object based on the provided configuration.
            If no configuration is provided (i.e., if preprocess_cfg is None), None is returned.
        """
        if preprocess_cfg is None: return None
        return Preprocessor(preprocess_cfg)

    @staticmethod
    def load_file_paths_from_dir(data_root: str, dataset_dir: List[str], recursive: bool=True) -> List[Path]:
        """
        Recursively loads all file paths from the given directory or directories.

        Args:
            data_root (str): The root directory to which the dataset directory paths are relative.
            dataset_dir (List[str]): A list of directory paths to load file paths from.
            recursive (bool, optional): Whether to recursively load file paths from all subdirectories. Defaults to True.

        Returns:
            List[Path]: A sorted list of all file paths found in the given directories.

        Raises:
            FileNotFoundError: If any of the given directories are not found.

        """
        file_list = []
        for dir in dataset_dir:
            dir = Path(data_root+dir)
            try:
                for dirname in dir.iterdir():
                    if dirname.is_file():
                        file_list.append(dirname)
                    elif dirname.is_dir() and recursive:
                        file_list += Pred_dataset.load_file_paths_from_dir('', [dirname.as_posix()], recursive=recursive)
            except FileNotFoundError as ex:
                logger.error(ex)
                logger.error(f"skipping dir: {dir}")

        return natsorted(file_list)
    

    def check_dataset_constrains(self):
        logger.warning("No dataset constrains for this Dataset! Please Check!")


    def __len__(self):
        return self.tiff_intersection_border[-1]
   
    def __getitem__(self, idx):
        """
        Returns a single sample of the dataset at the given index.

        Args:
            idx (int or Tensor): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the data_path and label_path (if applicable) of the sample.
            Tensor: The data for the sample, preprocessed and transformed.
            Tensor: The label for the sample (if applicable), preprocessed and transformed.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load Image
        raw_data = self.get_tile_from_index(idx)
        raw_data = raw_data.unsqueeze(dim=0)
        
        data = raw_data
        label = raw_data
        
        """
            UNCHANGED = 0
            GRAY = 1
            GRAY_ALPHA = 2
            RGB = 3
            RGB_ALPHA = 4
        """

        # Preprocess
        if self.preprocess:
            preprocess, label = self.preprocess(data, label)

        # Transform
        if self.transform:
            transform, label = self.transform(preprocess, label)

    
        return transform.float(), label.float() #, preprocess.float()