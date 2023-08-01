import imp
import multiprocessing

from numpy import block
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import time
from torchvision import transforms

import matplotlib.pyplot as plt

from BaseDataset import BaseDataset

class prepare_data:
    @staticmethod
    def prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, size, tile_grid):
        """ Prepare dataset for training and testing
        Args:
            data_root: root path of dataset
            dataset_dir: directory of dataset
            transform_cfgs: configurations for transform
            preprocess_cfgs: configurations for preprocess
        Returns:
            dataset object
        """
        dataset = BaseDataset('SR', 'train', size, tile_grid, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
        return dataset
    @staticmethod
    def prepare_dataloader(dataset, batch_size, split_factor=0.8, shuffle=True, num_workers=None):
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
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        length_dataset = dataset.__len__()
        train_size = int(split_factor * length_dataset)
        test_size = length_dataset - train_size

        train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return train_dataloader, test_dataloader
    
    @staticmethod
    def prepare_pred_dataset(data_root, dataset_dir, size, tile_grid, transform_cfgs=None, preprocess_cfgs=None, subset_indices: list = None):
        """ Prepare dataset for prediction
        Args:
            data_root: root path of dataset
            dataset_dir: directory of dataset
        Returns:
            dataset object
        """
        dataset = BaseDataset('SR', 'pred', size, tile_grid, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
        subset = torch.utils.data.Subset(dataset, subset_indices)
        return subset
    
    @staticmethod
    def plotimgs(input: list, pred: list, label:list, cmap: str = None, figsize: tuple = (20, 20)):
        """ Plot images
        Args:
            imgs: list of images
            titles: list of titles
            cmap: color map
            figsize: size of figure
        Returns:
            None
        """
        
        num_imgs_pred = len(pred)
        num_imgs_input = len(input)

        if num_imgs_pred == num_imgs_input:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            for i in range(num_imgs_input):
                axes[0].imshow(input[i], cmap=cmap)
                axes[0].set_title('input')

                axes[1].imshow(pred[i], cmap=cmap)
                axes[1].set_title('prediction')

                axes[2].imshow(label[i], cmap=cmap)
                axes[2].set_title('label')

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()
                time.sleep(0.5)
            
        
        

    @staticmethod
    def check_dataset(dataset):
        #imgs = []
        #titles = ['input', 'label']
        length = dataset.__len__()
        print(length)
        for i in range(length):
            input, label = dataset.__getitem__(i)

            print(i, input.shape, label.shape)
            input = input.squeeze(0).numpy()
            label = label.squeeze(0).numpy()

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(input, cmap='gray')
            axes[0].set_title('input')

            axes[1].imshow(label, cmap='gray')
            axes[1].set_title('label')

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            time.sleep(0.5)           

    def check_dataloader(dataloader):
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

