import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
from tqdm import tqdm

from Dataset import *
from BaseDataset import * 
from model.SRCNN import *
from model.Unet import * 
from image_utils.utils import * 
from unet import UNet

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
data_root = '/Users/haoruilong/BA_code/SR_for_CT_image_of_Batteries'
dataset_dir = ['/Dataset/Pristine']
outputs_path = '/Users/haoruilong/BA_code/SR_for_CT_image_of_Batteries/outputs'

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
    preprocess_cfgs = None

# Dataset for all, size: 256*256
mydataset = BaseDataset('SR', 'train', args.size, dataset_dir, data_root, None, preprocess_cfgs)

# Data splitting
length_dataset = mydataset.__len__()
train_size = int(0.8 * length_dataset)
test_size = length_dataset - train_size

train_dataset, test_dataset = data_utils.random_split(mydataset, [train_size, test_size])
# Apply train and test dataset in dataloader
mydataloader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

"""
# Test if dataloader successful is
print('length of train_dataset: ', train_dataset.__len__())
print('length of my dataset: ', length_dataset)

page_count = mydataset.get_pages_count_in_tile('./Dataset/Pristine/PTY_pristine_raw.tif')
print('total page count for each tiff file:', page_count)


for batch in mydataloader:
    inputs, labels = batch
    pass


"""
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
    
elif args.model == "Unet":
    model = Unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
plt.title('Training Loss for {}'.format(args.model))
plt.savefig('./outputs')