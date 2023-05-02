from Dataset import *
from model import *
import torch
import lightning as L

if __name__ == "__main__":
    inpath = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_raw"
    dm = MyDataModule(inpath, 32)
    model = MyModel(num_channels=1)

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, dm)
