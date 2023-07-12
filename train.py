import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import pytorch_lightning as L

from unet.ConvUNet import ConvUNet
from data_utils import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', type=str, default='SRCNN')
parser.add_argument('-s', '--size', dest='size', type = int, help='The size of each tile in pages in tiff', default=256)
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=1e-4)
parser.add_argument('-ep', '--num_epochs', dest='num_epochs', type=int, default=100)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('-bi', '--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
args = parser.parse_args()

# Prepare dataloader for training and testing
# Data path
data_root = f'H:\SR_for_CT_image_of_Batteries'
dataset_dir = [f'\dataset\pristine']

# Prepare configurations for dataset
cfgs_path_p = data_root + '/configs/preprocess.yaml'
cfgs_path_t = data_root + '/configs/transform.yaml'

if os.path.exists(cfgs_path_p):
    preprocess_cfgs = OmegaConf.load(cfgs_path_p)
else:
    preprocess_cfgs = None

if os.path.exists(cfgs_path_t):
    transform_cfgs = OmegaConf.load(cfgs_path_t)
else:
    transform_cfgs = None

dataset = prepare_data.prepare_dataset(data_root, dataset_dir, transform_cfgs, preprocess_cfgs, args.size)
train_dataloader, test_dataloader = prepare_data.prepare_dataloader(dataset, args.batch_size)

model = ConvUNet(image_channels=1, output_channels=1)
trainer = L.Trainer(max_epochs=args.num_epochs)
trainer.fit(model, train_dataloader)

'''
# Train the model
cudnn.benchmark = True
device = torch.device("mps")
# If the model is SRCNN
if args.model == "SRCNN":

    model = SRCNN().to(device)
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)
    

elif args.model == "UNet":
    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

else:
    raise NotImplementedError


criterion = nn.MSELoss()
best_weight = copy.deepcopy(model.state_dict())
best_epoch = 0
best_psnr = 0.0

for epoch in range(args.num_epochs):
    model.train()
    epoch_losses = AverageMeter()
    train_losses = []



    with tqdm(total=(train_dataset.__len__() - train_dataset.__len__() % args.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        
        for batch in train_dataloader:
            inputs, labels = batch

            
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))
            train_losses.append(epoch_losses.avg)

    torch.save(model.state_dict(), os.path.join(outputs_path, 'epoch_{}.pth'.format(epoch)))

    """
    model.eval()
    epoch_psnr = AverageMeter()

    for batch in test_dataloader:
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 0.1)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
    
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weight = copy.deepcopy(model.state_dict())
    """

    model.eval()
    eval_epoch_losses = AverageMeter()

    for batch in tqdm(test_dataloader, desc='eval...'):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 0.1)

        eval_loss = criterion(preds, labels)

        eval_epoch_losses.update(eval_loss.item(), len(inputs))

    print('eval psnr: {:.2f}'.format(eval_epoch_losses.avg))

    if eval_epoch_losses.avg > best_psnr:
        best_epoch = epoch
        best_psnr = eval_epoch_losses.avg
        best_weight = copy.deepcopy(model.state_dict()) 


print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weight, os.path.join(outputs_path, 'best.pth'))

plt.plot(train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss for {} with binning factor: {}'.format(args.model, args.binning_factor))
plt.savefig('./outputs/model_{}.png'.format(args.model))
'''