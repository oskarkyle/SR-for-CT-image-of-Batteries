from loguru import logger
import random
from typing import Dict, Any, Tuple, Optional
from omegaconf import DictConfig, ListConfig
import torch
import kornia
import torchvision.transforms.functional as F
import cv2
import numpy as np
import math
from PIL import Image

# https://kornia.readthedocs.io/en/latest/augmentation.html
class Preprocessor:
    def __init__(self, preprocess_cfg: DictConfig):
        self.preprocess_cfg = preprocess_cfg
        self.preprocess_cfg.convert_dtype = None
        self.preprocess_map = {
            "resize": {"func": self.resize, "mask": True},
            "rescale": {"func": self.rescale, "mask": True},
            "normalize": {"func": self.normalize, "mask": False},
            "gamma_correction": {"func": self.gamma_correction, "mask": False},
            "binning": {"func": self.binning, "mask": False},
            "to_grayscale": {"func": self.to_grayscale, "mask": True},
            "convert_dtype": {"func": self.convert_dtype, "mask": True},
        }

    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        # Apply relevant preprocessing operations to image and mask
        for preprocess_name, preprocess_param in self.preprocess_cfg.items():
            preprocessor_func = self.preprocess_map.get(preprocess_name)['func']
            for_mask = self.preprocess_map.get(preprocess_name)['mask']
            if preprocessor_func is not None:
                # Ensure input tensor has shape (C, H, W) or (B, C, H, W)
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                if preprocess_param is None:
                    image = preprocessor_func(image)
                else:
                    image = preprocessor_func(image, **preprocess_param)
                # Remove the batch dimension if present
                if image.shape[0] == 1:
                    image = image.squeeze(0)
                if mask is not None and for_mask:
                    # Ensure input tensor has shape (C, H, W) or (B, C, H, W)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                    if preprocess_param is None:
                        mask = preprocessor_func(mask)
                    else:
                        mask = preprocessor_func(mask, **preprocess_param)
                    # Remove the batch dimension if present
                    if mask.shape[0] == 1:
                        mask = mask.squeeze(0)
            else:
                logger.exception(f"Invalid preprocessor operation: {preprocess_name}")
                # raise ValueError("Invalid preprocessor operation.")

        return image, mask

    @staticmethod
    def normalize(image: torch.Tensor, mean: list, std: list) -> torch.Tensor:
        """
        Normalize the image tensor with given mean and standard deviation values.
        """
        mean = torch.tensor(mean, device=image.device)
        std = torch.tensor(std, device=image.device)

        norm_img = kornia.enhance.normalize(image, mean=mean, std=std)

        return norm_img

    @staticmethod
    def rescale(image: torch.Tensor, min_val: int, max_val: int) -> torch.Tensor:
        """
        :param image:
        :param min_val:
        :param max_val:
        :return: rescaled image
        """
        if image.max() > 1:
            image = image/255

        return image*(max_val-min_val)+min_val

    @staticmethod
    def resize(image: torch.Tensor, size: list[int, int], padding=True) -> torch.Tensor:
        """
        Resize a torch.Tensor image to the specified size, while padding it with zeros to make it square, using Kornia library.

        Args:
            image (torch.Tensor): Input image tensor.
            size (list[int, int]): Desired output size.

        Returns:
            torch.Tensor: Resized and padded image tensor.
        """
        # Pad the image
        if padding:
            # Compute padding
            h, w = image.shape[-2], image.shape[-1]
            if h > w:
                pad_left = pad_right = (h - w) // 2
                pad_top = pad_bottom = 0
            else:
                pad_top = pad_bottom = (w - h) // 2
                pad_left = pad_right = 0
            padding = [pad_left, pad_right, pad_top, pad_bottom]

            #padded_img = kornia.geometry.transform.pad(img, padding, 'constant', 0)
            image = torch.nn.functional.pad(image, padding, mode='constant', value=0)

        resized_img = kornia.geometry.transform.resize(image, tuple(size), interpolation='bilinear')

        return resized_img

    @staticmethod
    def to_grayscale(image: torch.Tensor) -> torch.Tensor:
        """
        Convert the image tensor to grayscale.
        """
        if not image.shape[1] == 1:
            image = kornia.color.rgb_to_grayscale(image)

        return image

    @staticmethod
    def convert_dtype(image: torch.Tensor) -> torch.Tensor:
        """
        Convert the image tensor to the right dytpe.
        :param image:
        :return: image
        """
        if image.max() > 1:
            image = image/255

        return image

    @staticmethod
    def gamma_correction(image: torch.Tensor) -> torch.Tensor:
        # convert img to HSV
        image = np.transpose(image.numpy(), (0, 2, 3, 1))
        hsv = torch.from_numpy(cv2.cvtColor(image[0], cv2.COLOR_RGB2HSV))
        val = hsv[..., 2]

        # compute gamma = log(mid*255)/log(mean)
        mid = torch.tensor(0.5)
        mean = torch.mean(val.float())
        gamma = torch.log(mid * 255) / torch.log(mean)

        # do gamma correction on value channel
        val_gamma = torch.pow(val, gamma).clip(0, 255).to(torch.uint8)

        # combine new value channel with original hue and sat channels
        hsv_gamma = hsv.clone()
        hsv_gamma[..., 2] = val_gamma
        img_gamma = cv2.cvtColor(hsv_gamma.numpy(), cv2.COLOR_HSV2RGB)
        return torch.from_numpy(np.array([np.transpose(img_gamma, (2, 0, 1))]))

    @staticmethod
    def binning(image: torch.Tensor, binning_factor: int) -> torch.Tensor:
        if binning_factor < 2:
            return image

        shape = image.shape[-2:]

        binned = torch.nn.functional.avg_pool2d(image.float(), binning_factor, stride=binning_factor)

        resized = kornia.geometry.transform.resize(binned, tuple(shape), interpolation='nearest')

        return resized
    '''
    @staticmethod
    def binning(image, binning_factor, size):
        mat = np.array(image)
        mat_reshape = mat.reshape(mat.shape[0]//binning_factor, 2, mat.shape[1]//binning_factor, 2)
        bin_mat = np.mean(mat_reshape, axis=(1, 3))
    
        resize = cv2.resize(bin_mat, (size, size), interpolation=cv2.INTER_CUBIC)

        return resize
    '''