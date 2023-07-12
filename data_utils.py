import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from BaseDataset import BaseDataset

class prepare_data:
    @staticmethod
    def prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, size):
        """ Prepare dataset for training and testing
        Args:
            data_root: root path of dataset
            dataset_dir: directory of dataset
            transform_cfgs: configurations for transform
            preprocess_cfgs: configurations for preprocess
        Returns:
            dataset object
        """
        dataset = BaseDataset('SR', 'train', size, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
        return dataset
    @staticmethod
    def prepare_dataloader(dataset, batch_size, split_factor=0.8, shuffle=True, num_workers=4):
        """ Prepare dataloader for training and testing
        Args:
            dataset: dataset object
            batch_size: batch size
            split_factor: split factor for train and test
            shuffle: shuffle or not
            num_workers: number of workers
        Returns:
            train and test dataloader
        """
        length_dataset = dataset.__len__()
        train_size = int(split_factor * length_dataset)
        test_size = length_dataset - train_size

        train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return train_dataloader, test_dataloader