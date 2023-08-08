import pytorch_lightning as L


from source.BaseDataset import *
from source.data_utils import *
from model.unet.ConvUNet import *
from utilities import *



if __name__ == "__main__":
    # config's path
    train_cfg = r'.\configs\train.yaml'
    preprocess_cfgs = r'.\configs\preprocess.yaml'

    # Data
    subset, pred_loader = pred_data(train_cfg, preprocess_cfgs)


    # Load model

    ckpt_path = r'.\lightning_logs\version_5\checkpoints\epoch=0-step=9280.ckpt'

    model: L.LightningModule = ConvUNet().restore(ckpt_path)
    print(is_lightning_module(model))

    # Predict
    trainer = L.Trainer()
    predictions = trainer.predict(model, pred_loader)

    # Plot
    plot_img(predictions, pred_loader)
    