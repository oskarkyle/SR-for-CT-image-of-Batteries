import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import pytorch_lightning as L
from omegaconf import DictConfig, ListConfig, OmegaConf
from loguru import logger
import matplotlib.pyplot as plt
import time
from torchvision import transforms

from BaseDataset import BaseDataset
from data_utils import prepare_data
from unet.ConvUNet import ConvUNet

def load_data(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, size, batch_size, subset_indices: list = None):
    pred_dataset = prepare_data.prepare_pred_dataset(data_root, dataset_dir, size, transform_cfgs, preprocess_cfgs, subset_indices)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return pred_dataset, pred_dataloader


def restore(cls: L.LightningModule, ckpt: str):#, map_location = torch.device("cpu")):
        """
        Restores a PyTorch Lightning model from a checkpoint file.

        Args:
            cls (LightningModule): The class of the PyTorch Lightning model to restore.
            ckpt (str): The path to the checkpoint file to restore from.
            map_location (Union[torch.device, str, int]): Device to map the restored model to.

        Returns:
            Tuple[LightningModule, DictConfig]: A tuple of the restored model and its configuration.

        Raises:
            RuntimeError: If the checkpoint file does not contain hyperparameters.

        Example:
            # Restore a PLModel from a checkpoint file
            model, config = PLModel.restore(ckpt='path/to/checkpoint.ckpt')
        """
        torch_ckpt = torch.load(ckpt)
        if "hyper_parameters" not in torch_ckpt:
            logger.error("Checkpoint does not contain hyperparameters.")
            raise RuntimeError("Checkpoint does not contain hyperparameters.")

        logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {cls.__name__}\n\tcheckpoint: {ckpt}")
        model = cls.load_from_checkpoint(checkpoint_path=ckpt)#, map_location=map_location)
        logger.success(f"Successfully loaded checkpoint")

        return model#, OmegaConf.create(torch_ckpt["hyper_parameters"])

def prediction(model, dataloader):

    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        print(outputs.shape)

        lenght = outputs.shape[0]
        for i in range(lenght):
            output = outputs[i, :, :, :]
            output = output.squeeze(0)
            output = output.detach().numpy()
            plt.imshow(output, cmap='gray')
            plt.show()
            
             
            
            

if __name__ == "__main__":

    data_root = "/Users/haoruilong/BA_code/SR_for_CT_image_of_Batteries"
    dataset_dir = "/Dataset/Pristine"

    cfgs_path_p = data_root + '/configs/preprocess.yaml'
    preprocess_cfgs = OmegaConf.load(cfgs_path_p)

    batch_size = 4
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    pred_dataset, pred_dataloader = load_data(data_root, dataset_dir, None, preprocess_cfgs, 512, batch_size, subset_indices)
    #prepare_data.check_dataset(pred_dataset)

    ckpt_path = "/Users/haoruilong/BA_code/version_1/checkpoints/epoch=0-step=9280.ckpt"

    model = ConvUNet(image_channels=1, output_channels=1)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    model_state_dict = checkpoint['state_dict']
    trained = model.load_state_dict(model_state_dict)
    
    prediction(model, pred_dataloader)

