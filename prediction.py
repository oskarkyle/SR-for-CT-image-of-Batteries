import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import pytorch_lightning as L

from BaseDataset import BaseDataset
from data_utils import prepare_data
from unet.ConvUNet import ConvUNet

def load_data(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, size, batch_size, subset_indices: list = None):
    pred_dataset = prepare_data.prepare_pred_dataset(data_root, dataset_dir, size, transform_cfgs, preprocess_cfgs, subset_indices)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return pred_dataloader

def predict(model_path, pred_dataloader):
    model = ConvUNet.load_from_checkpoint(model_path)
    trainer = L.Trainer()
    predictions = trainer.predict(model, pred_dataloader)
    return predictions