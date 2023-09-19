import matplotlib.pyplot as plt
from torch.utils.data import random_split
from natsort import natsorted
from loguru import logger
from source.BaseDataset import *
from image_utils.utils import *
from torch.utils.data import Dataset


def split_dataset(cfg: DictConfig, dataset):
    split_ratios = cfg.split_ratios
    train_ratio = split_ratios.train
    val_ratio = split_ratios.val 
    test_ratio = split_ratios.test

    dataset_size = dataset.__len__()
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    train_dataset,validation_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])
    return train_dataset,validation_dataset,test_dataset

def save_image(image_list: list, output_path, title_list: list = None, axis: bool = False):
    fig, ax = plt.subplots(1, len(image_list), figsize=(12, 5))
    for i in range(len(image_list)):
        ax[i].imshow(convert_tensor_to_numpy(image_list[i]), cmap='gray')
        if not axis:
            ax[i].axis('off')
        if title_list:
            title = ax[i].set_title(title_list[i])
            title.set_fontsize(12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')

def save_image_in_RGB(input: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, output_path: str):
    input = convert_tensor_to_numpy(input)
    pred = convert_tensor_to_numpy(pred)
    label = convert_tensor_to_numpy(label)

    input_rgb = convert_gray_to_rgb(input)
    pred_rgb = convert_gray_to_rgb(pred)
    label_rgb = convert_gray_to_rgb(label)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(input_rgb)
    title = ax[0].set_title('Input')
    title.set_fontsize(20)

    ax[1].imshow(pred_rgb)
    title = ax[1].set_title('Prediction')
    title.set_fontsize(20)

    ax[2].imshow(label_rgb)
    title = ax[2].set_title('Label')
    title.set_fontsize(20)

    
    plt.tight_layout()
    plt.savefig(output_path)

def setup_pred_tiles_and_labels(dataset: BaseDataset):
    tiles = []
    labels = []

    for i in range(dataset.__len__()):
        input, label = dataset.__getitem__(i)
        tiles.append(input)
        labels.append(label)
    
    return tiles, labels

def calc_total_page_of_one_file(cfg: DictConfig, dataset: BaseDataset):
    total_tiles_in_one_page = cfg.dataset.tile_grid ** 2
    file_list = BaseDataset.load_file_paths_from_dir(cfg.dataset.data_root, cfg.dataset.dataset_dir)
    total_files = len(file_list)
    
    dataset_size = dataset.__len__()
    total_page = dataset_size // total_tiles_in_one_page
    total_page_of_one_file = total_page // total_files
    return total_page_of_one_file

class TiffDataset(Dataset):
    def __init__(self, 
                 cfg: DictConfig,
                 ):   
        self.data_root = cfg.data_root
        self.dataset_dir = cfg.dataset_dir
        self.task = cfg.task
        self.cfg = cfg
        self.file_list = self.load_file_paths_from_dir(self.data_root, self.dataset_dir)
        self.process_cfg = cfg.preprocess_cfg
        self.preprocess = self.get_preprocessor(self.process_cfg)
        self.transform_cfg = cfg.transforms_cfg
        self.transform = self.get_transform(self.transform_cfg)

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
        
        if preprocess_cfg is None:
            return None
        return Preprocessor(preprocess_cfg)
    
    @staticmethod
    def get_transform(transforms_cfg: DictConfig) -> Optional[Transforms]:
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
    
    def load_page_from_tiff_file(self):
        file_list = self.file_list
        pages_tensor = []
        for i in range(len(file_list)):
            tiff_file = file_list[i]
            with tifffile.TiffFile(tiff_file) as tif:
                pages_number = len(tif.pages)
                for i in range(pages_number):
                    page = tif.pages[i]
                    page_array = page.asarray()
                    page_tensor = torch.from_numpy(page_array)
                    pages_tensor.append(page_tensor)
                
                return pages_tensor
    
    def __len__(self):
        pages_tensor = self.load_page_from_tiff_file()
        return len(pages_tensor)
    
    def __getitem__(self, index):
        pages_tensor = self.load_page_from_tiff_file()
        raw_data_tensor = pages_tensor[index]
        raw_data_float_tensor = torch.tensor(raw_data_tensor.clone().detach(), dtype=torch.float32)
        data = raw_data_float_tensor.unsqueeze(0)
        label = data

        #Preprocess
        if self.preprocess:
            data, label = self.preprocess(data, label)

        # Transform
        if self.transform:
            data, label = self.transform(data, label)

        return data, label
    
def average_psnr(psnr_list: list):
    psnr_average = sum(psnr_list) / len(psnr_list)
    print(psnr_average)