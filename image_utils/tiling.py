import torch
from image_utils.utils import *

class Tile():
    def __init__(self,sequence_number,pic_identifier,tile,h_tiles,w_tiles,tile_size):
        self.sequence_number = sequence_number
        self.pic_identifier = pic_identifier
        self.tile = tile    
        self.tiling_shape = (h_tiles,w_tiles)       
        self.tile_size = tile_size

        # Determine if the tile is fully sized
        if(tile.shape == (tile_size,tile_size)):
             self.sparsed = False
             #print("Fully sized tile")
        else:
             self.sparsed = True
             #print("Sparsed tile")
    
    def shape(self):
        return self.tile.shape
    
    def check_sparsed(self):
         return self.sparsed
    
    def get_image(self):
         return self.tile
    
    def get_pic_identifier(self):
         return self.pic_identifier
    
    def get_sequence_number(self):
         return self.sequence_number
    
    def pad(self):
        img = self.tile
        expected_shape = [self.tile_size,self.tile_size]
        current_shape = img.shape
        if current_shape == expected_shape:
            return img
        else:
             pad_width = (0, expected_shape[1] - img.shape[1] , 0 , expected_shape[0] - img.shape[0])
             new_img = torch.nn.functional.pad(img, pad_width, mode='constant', value=0)
             return new_img
    
    def list_info(self):
         print("----------------------------list_info--------------------------------\n")
         print("shape: {}, pic_indentifier:{}, sequence_number:{}  ".format(self.shape(), self.pic_identifier,self.sequence_number))
    
# Receives a numpy array and returns a list of Tile objects
# Todo: Optimization for native numpy return value

def get_tiling_grid(tile_grid):#image,tile_grid):
     '''height, width = image.shape[:2]
  
     tiles_count_vertical = height // tile_size
     h_rest = height % tile_size
     if(h_rest != 0):
        tiles_count_vertical += 1


     tiles_count_horizontal = width // tile_size
     w_rest = width % tile_size
     if(w_rest != 0):
        tiles_count_horizontal += 1

     return [tiles_count_vertical,tiles_count_horizontal]'''

     return [tile_grid,tile_grid]

def pad(tile,tile_size):
    img = tile
    expected_shape = [tile_size,tile_size]
    current_shape = img.shape
    if current_shape == expected_shape:
        return img
    else:
        pad_width = (0, expected_shape[1] - img.shape[1] , 0 , expected_shape[0] - img.shape[0])
        new_img = torch.nn.functional.pad(img, pad_width, mode='constant', value=0)
        return new_img

def split(image, tile_size,pic_identifier):
    height, width = image.shape[:2]
    h_tiles,w_tiles = get_tiling_grid(image,tile_size)
    tiles = []

    for h in range(h_tiles):
        if h != h_tiles-1:
            for w in range(w_tiles):
                    if w != w_tiles-1:
                        raw_tile = image[h*tile_size:(h+1)*tile_size, w*tile_size:(w+1)*tile_size]
                        #print("tile_image shape:", raw_tile.shape)
                        tile = Tile( h*w_tiles+w+1,pic_identifier,raw_tile,h_tiles,w_tiles,tile_size)
                        tiles.append(tile)
                    else:
                        raw_tile = image[h*tile_size:(h+1)*tile_size, w*tile_size:width]
                        tile = Tile( h*w_tiles+w+1,pic_identifier,raw_tile,h_tiles,w_tiles,tile_size)
                        #print("tile_image shape:", raw_tile.shape)
                        tiles.append(tile)
        else:
            for w in range(w_tiles):
                    if w != w_tiles-1:
                        raw_tile = image[h*tile_size:height, w*tile_size:(w+1)*tile_size]
                        tile = Tile( h*w_tiles+w+1,pic_identifier,raw_tile,h_tiles,w_tiles,tile_size)
                        #print("tile_image shape:", raw_tile.shape)
                        tiles.append(tile)
                    else:
                        raw_tile = image[h*tile_size:height, w*tile_size:width]
                        tile = Tile( h*w_tiles+w+1,pic_identifier,raw_tile,h_tiles,w_tiles,tile_size)
                        #print("tile_image shape:", raw_tile.shape)
                        tiles.append(tile)
             
        
    return tiles

# Fetch and return a tile on its sequence number in original pic 
def get_tile_by_sequence_number(image,sequence_number,tile_grid: list): #tile_size,
     height, width = image.shape[:2]
     #h_tiles,w_tiles = get_tiling_grid(tile_grid
     h_tiles = tile_grid[0]
     w_tiles = tile_grid[1]

     vertical_size = height // h_tiles
     horizontal_size = width // w_tiles
     # print("tiles:",h_tiles,w_tiles)
     position_h = sequence_number // h_tiles
     position_w = sequence_number % w_tiles
     
     if(position_h < h_tiles - 1 and position_w < w_tiles - 1):
        # print("here")
        raw_tile = image[position_h*vertical_size:(position_h + 1)*vertical_size, position_w*horizontal_size:(position_w+1)*horizontal_size]
        return raw_tile
     elif(position_h == h_tiles - 1 and position_w < w_tiles -1):
        raw_tile = image[position_h*vertical_size:height, position_w*horizontal_size:(position_w+1)*horizontal_size]
        return raw_tile
     elif(position_h < h_tiles - 1 and position_w == w_tiles -1):
        raw_tile = image[position_h*vertical_size:(position_h+1)*vertical_size, position_w*horizontal_size:width]
        return raw_tile
     elif(position_h == h_tiles - 1 and position_w == w_tiles -1):
        raw_tile = image[position_h*vertical_size:height, position_w*horizontal_size:width]
        return raw_tile
     else:
         print("position_h: {}, position_w: {}".format(position_h,position_w))
         raise ValueError
     

