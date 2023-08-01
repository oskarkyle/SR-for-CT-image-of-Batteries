from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Tuple, Optional, List, Union
import torch
from pathlib import Path
from collections import defaultdict
from natsort import natsorted

from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset as DS
from torch.utils.data import DataLoader
from transform.Transforms import Transforms
from preprocess.Preprocessor import Preprocessor

import tifffile
from PIL import Image
from image_utils import tiling
import matplotlib.pyplot as plt
import os
import time


MODE_DICT = {
    0: ImageReadMode.UNCHANGED,
    1: ImageReadMode.GRAY,
    2: ImageReadMode.GRAY_ALPHA,
    3: ImageReadMode.RGB,
    4: ImageReadMode.RGB_ALPHA
}


class BaseDataset(DS):
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
        self.transform_cfg = transforms_cfg
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
                        file_list += BaseDataset.load_file_paths_from_dir('', [dirname.as_posix()], recursive=recursive)
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
            data, label = self.preprocess(data, label)

        # Transform
        if self.transform:
            data, label = self.transform(data, label)

    
        return data.float(), label.float()


if __name__ == "__main__":
    data_root = r'H:\SR_for_CT_image_of_Batteries'
    dataset_dir = [r'\dataset\pristine']

    cfgs_path_p = data_root + '\configs\preprocess.yaml'
    cfgs_path_t = data_root + '\configs\transform.yaml'

    if os.path.exists(cfgs_path_p):
        preprocess_cfgs = OmegaConf.load(cfgs_path_p)
    else:
        preprocess_cfgs = None

    if os.path.exists(cfgs_path_t):
        transform_cfgs = OmegaConf.load(cfgs_path_t)
    else:
        preprocess_cfgs = None

    filelist = BaseDataset.load_file_paths_from_dir(data_root, dataset_dir)
    print(filelist)

    mydataset = BaseDataset('SR', 'train', 512, 4, dataset_dir, data_root, None, preprocess_cfgs)
    
    
    length_dataset = mydataset.__len__()
    print(length_dataset)
    for i in range(length_dataset):
        '''
        data, label = mydataset.__getitem__(i)
        data = data.squeeze().numpy()
        label = label.squeeze().numpy()

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data, cmap='gray')
        ax[0].set_title('input')
        ax[1].imshow(label, cmap='gray')
        ax[1].set_title('label')

        plt.tight_layout()
        plt.show()
        '''
        data, label = mydataset.__getitem__(i)