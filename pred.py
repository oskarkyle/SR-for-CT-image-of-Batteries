import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning import Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset
from PIL import Image
import kornia

from source.BaseDataset import *
from utilities import *
from model.unet.ConvUNet import *
from train import *
from image_utils.utils import *
from preprocess.Preprocessor import *

def setup_test_set(cfg: DictConfig):
    tiff_dataset = TiffDataset(cfg.dataset)
    train_set, val_set, test_set = split_dataset(cfg, tiff_dataset)
    return test_set

def setup_input_label(cfg: DictConfig):
    test_set = setup_test_set(cfg)
    input, label = test_set.__getitem__(cfg.pred.slice - 1)
    return input, label

def splitting(image: torch.Tensor, cfg: DictConfig):
    image = convert_tensor_to_numpy(image)

    # Get the dimensions of the image
    height, width = image.shape

    # Define the row and column 
    row = cfg.dataset.tile_grid
    col = cfg.dataset.tile_grid

    # Define tile size
    tile_height = height // col
    tile_width = width // row   

    # Initialize an empty list to store the crops
    tiles = []

    # Iterate over the rows and columns
    for i in range(col):
        for j in range(row):
            # Calculate the coordinates for the current crop
            top = i * tile_height
            bottom = (i + 1) * tile_height
            left = j * tile_width
            right = (j + 1) * tile_width

            # split the image
            tile = image[top:bottom, left:right]
            tile = torch.from_numpy(tile)
            if tile.shape != (cfg.dataset.tile_size, cfg.dataset.tile_size):
                tile = Preprocessor.resize(tile, (cfg.dataset.tile_size, cfg.dataset.tile_size))
            
            tile = tile.unsqueeze(0)
            
            # Append the crop to the list
            tiles.append(tile)

    return tiles

def load_model(cfg: DictConfig):
    path = cfg.dataset.data_root
    os.chdir(path)
    myckpt_path = os.getcwd() + cfg.pred.ckpt_path
    logger.info(f"Attempting to load checkpoint .. \n\tmodel_class: {ConvUNet.__name__}\n\tcheckpoint: {myckpt_path}")
    model = ConvUNet.load_from_checkpoint(myckpt_path, map_location=torch.device(cfg.pred.device), ch_mults=cfg.model.ch_mults, n_blocks=cfg.model.n_blocks, n_layers=cfg.model.n_layers)
    logger.success(f"Successfully loaded checkpoint")

    return model


def prediction(model, tile):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    input = tile.to(device)
    output = model(input)
    return output

def reassemble(tiles: list, cfg:DictConfig):
    reassembled_image = np.zeros((cfg.dataset.tile_size * cfg.dataset.tile_grid, cfg.dataset.tile_size * cfg.dataset.tile_grid), dtype=tiles[0].dtype)
    num_col = cfg.dataset.tile_grid
    num_row = cfg.dataset.tile_grid
    for i in range(num_col):
        for j in range(num_row):
            y_start = i * cfg.dataset.tile_size
            y_end = (i + 1) * cfg.dataset.tile_size

            x_start = j * cfg.dataset.tile_size
            x_end = (j + 1) * cfg.dataset.tile_size

            reassembled_image[y_start:y_end, x_start:x_end] = tiles[i * num_col + j]

    reassembled_image = torch.from_numpy(reassembled_image)
    reassembled_image = Preprocessor.resize(reassembled_image, (cfg.interpolation.height, cfg.interpolation.width))
    return reassembled_image


def inference(cfg: DictConfig):
    model = load_model(cfg)
    model.eval()
    
    input, label = setup_input_label(cfg)
    input_resize = Preprocessor.resize(input, (cfg.dataset.tile_size, cfg.dataset.tile_size))

    inputs = splitting(input, cfg)
    labels = splitting(label, cfg)
    inputs_resize = splitting(input_resize, cfg)

    for i in range(len(inputs)):
        input = inputs[i].unsqueeze(0)
        label = labels[i].unsqueeze(0)
        input_resize = inputs_resize[i].unsqueeze(0)
        output = prediction(model, input)
        output = output.cpu()

        
        print(f"tile_{i + 1}_output_shape: {output.shape}")

        images = [input, output, label]
        titles = ['LR', 'SR', 'Ground Truth']
        output_path = os.getcwd() + cfg.pred.output_dir + cfg.pred.output_name + f"_slice_" + str(cfg.pred.slice) + f"_{i}.png"

        save_image(images, output_path, titles)
    
    #save_image([input_resize, input, label], os.getcwd()+cfg.pred.output_dir+cfg.pred.output_name+'slice'+str(cfg.pred.slice)+'.png', ['LR', 'SR', 'Label'], axis=False)


def calc_average_psnr_in_testset(cfg:DictConfig):
    testset = setup_test_set(cfg)
    model = load_model(cfg)
    model.eval()
    psnr_list = []
    average_psnr_list = []

    for i in range(testset.__len__()):
        input, label = testset.__getitem__(i)
        split_inputs = splitting(input, cfg)
        split_labels = splitting(label, cfg)

        for j in range(len(split_inputs)):
            tile = split_inputs[j].unsqueeze(0)
            label = split_labels[j].unsqueeze(0)
            output = prediction(model, tile)
            output = output.cpu()
            psnr = calc_psnr(output, label)
            psnr = psnr.item()
            psnr_list.append(psnr)

        slice_psnr = sum(psnr_list) / len(psnr_list)
        print(f"第{i+1}张的PSNR: {slice_psnr}")
        average_psnr_list.append(slice_psnr)
    
    average_psnr(average_psnr_list)



