from loguru import logger
import random
from typing import Dict, Any, Tuple, Optional
from kornia.augmentation.base import _AugmentationBase
from omegaconf import DictConfig
import torch
import kornia
import torchvision.transforms.functional as F


# https://pytorch.org/vision/stable/transforms.html#functional-transforms
# https://kornia.readthedocs.io/en/latest/augmentation.html
class Transforms:
    """
    Apply a set of custom transforms on the input image

    Example:
        transform_dict = {
        'rotate': {'angle': 30},
        'adjust_brightness': {'brightness_factor': 0.5}
        }

    """

    def __init__(self, transforms_cfg: DictConfig):
        self.p = transforms_cfg.propability # transforms_cfg.pop('propability')
        self.transforms_cfg = transforms_cfg

        self.transforms_map = {
            "RandomAffine": {"func": self.RandomAffine, "mask": True, "channel": None},
            "RandomCrop": {"func": self.RandomCrop, "mask": True, "channel": '*'},
            "RandomHorizontalFlip": {"func": self.RandomHorizontalFlip, "mask": True, "channel": None},
            "RandomVerticalFlip": {"func": self.RandomVerticalFlip, "mask": True, "channel": None},
            "RandomBrightness": {"func": self.RandomBrightness, "mask": False, "channel": None},
            "RandomContrast": {"func": self.RandomContrast, "mask": False, "channel": None},
            "RandomSaturation": {"func": self.RandomSaturation, "mask": False, "channel": 3},
            "RandomHue": {"func": self.RandomHue, "mask": False, "channel": 3},
            #"RandomMedianBlur": {"func": self.RandomMedianBlur, "mask": False, "channel": None}, # scheint nicht mehr da zu sein
            "RandomGaussianNoise": {"func": self.RandomGaussianNoise, "mask": False, "channel": None},
            "RandomGaussianBlur": {"func": self.RandomGaussianBlur, "mask": False, "channel": None},
        }

    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:

        if random.random() > self.p:
            return image, mask

        # Apply relevant transform operations to image and mask
        for transform_name, transform_param in self.transforms_cfg.items():
            if transform_name == 'propability': 
                continue
            if self.transforms_map.get(transform_name)['channel'] and not self.transforms_map.get(transform_name)['channel'] == image.shape[0]: 
                continue # skip trafos that need right channels
            transforms_func: _AugmentationBase = self.transforms_map.get(transform_name)['func']
            for_mask = self.transforms_map.get(transform_name)['mask']
            if transform_name is not None: # ------------------------------------- for image
                aug = transforms_func(transform_param)
                # Ensure input tensor has shape (C, H, W) or (B, C, H, W)
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image = torch.clip(aug(image.float()), min=0.0, max=1.0)
                # Remove the batch dimension if present
                if image.shape[0] == 1:
                    image = image.squeeze(0)

                if mask is not None and for_mask: # ------------------------------ for mask
                    # Ensure input tensor has shape (C, H, W) or (B, C, H, W)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                    mask = aug(mask, params=aug._params)
                    # Remove the batch dimension if present
                    if mask.shape[0] == 1:
                        mask = mask.squeeze(0)
            else:
                logger.exception(f"Invalid transform operation: {transform_name}")
                #raise ValueError("Invalid transform operation.")

        return image, mask

    @staticmethod
    def RandomAffine(param):
        """
        Rotate the given tensor image by angle.

        Args:
            degrees (float or int): Rotation angle in degrees.
            etc.

        Returns:
            torch.Tensor: Rotated tensor image.
        """
        return kornia.augmentation.RandomAffine(**param)

    @staticmethod
    def RandomCrop(param):
        """
        Crop the given tensor image at a random location.

        Args:
            image (torch.Tensor): Tensor image to be cropped.
            output_size (tuple): Expected output size (height, width) of the crop.

        Returns:
            torch.Tensor: Cropped tensor image.
        """
        return kornia.augmentation.RandomCrop(**param)

    @staticmethod
    def RandomHorizontalFlip(param):
        """
        Horizontally flip the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be flipped.

        Returns:
            torch.Tensor: Horizontally flipped tensor image.
        """
        return kornia.augmentation.RandomHorizontalFlip(**param)

    @staticmethod
    def RandomVerticalFlip(param):
        """
        Vertically flip the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be flipped.

        Returns:
            torch.Tensor: Vertically flipped tensor image.
        """
        return kornia.augmentation.RandomVerticalFlip(**param)

    @staticmethod
    def RandomBrightness(param):
        """
        Adjust the brightness of the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be adjusted.
            brightness_factor (float): Brightness adjustment factor. Must be positive.

        Returns:
            torch.Tensor: Brightness-adjusted tensor image.
        """
        return kornia.augmentation.RandomBrightness(**param)

    @staticmethod
    def RandomContrast(param):
        """
        Adjust the contrast of the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be adjusted.
            contrast_factor (float): Contrast adjustment factor. Must be positive.

        Returns:
            torch.Tensor: Contrast-adjusted tensor image.
        """
        return kornia.augmentation.RandomContrast(**param)

    @staticmethod
    def RandomSaturation(param):
        """
        Adjust the saturation of the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be adjusted.
            saturation_factor (float): Saturation adjustment factor. Must be positive.

        Returns:
            torch.Tensor: Saturation-adjusted tensor image.
        """
        return kornia.augmentation.RandomSaturation(**param)

    @staticmethod
    def RandomHue(param):
        """
        Adjust the hue of the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be adjusted.
            hue_factor (float): Hue adjustment factor.

        Returns:
            torch.Tensor: Hue-adjusted tensor image.
        """
        return kornia.augmentation.RandomHue(**param)

    @staticmethod
    def RandomMedianBlur(param):
        """
        Applies a random median blur to the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be blurred.
            kernel_size_range (tuple): Range of kernel sizes to randomly sample from.

        Returns:
            torch.Tensor: Blurred tensor image.
        """
        kernel_size_range = param['kernel_size_range']
        kernel_size = torch.randint(kernel_size_range[0], kernel_size_range[1] + 1, (1,))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kornia.augmentation.RandomMedianBlur(kernel_size.item())

    @staticmethod
    def RandomGaussianNoise(param):
        """
        Adds random Gaussian noise to the given tensor image. If values are out of range, the tensor is clipped.

        Args:
            image (torch.Tensor): Tensor image to be noised.
            mean (float): Mean of the Gaussian noise distribution.
            std (float): Standard deviation of the Gaussian noise distribution.

        Returns:
            torch.Tensor: Noised tensor image.
        """
        return kornia.augmentation.RandomGaussianNoise(**param)

    @staticmethod
    def RandomGaussianBlur(param):
        """
        Applies a random Gaussian blur to the given tensor image.

        Args:
            image (torch.Tensor): Tensor image to be blurred.
            kernel_size_range (tuple): Range of kernel sizes to randomly sample from.
            sigma_range (tuple): Range of sigma values to randomly sample from.

        Returns:
            torch.Tensor: Blurred tensor image.
        """
        kernel_size_range = param['kernel_size_range']
        sigma_range = param['sigma_range']
        kernel_size = torch.randint(kernel_size_range[0], kernel_size_range[1] + 1, (1,))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kornia.augmentation.RandomGaussianBlur(kernel_size.item(), sigma_range)

