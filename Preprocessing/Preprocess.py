import pytiff
import numpy as np
from PIL import Image
import cv2
import glob
from Special_binning import *
from Dataset import *

# Load data
def load_data(path):
    images = []
    with pytiff.Tiff(path) as handle:
        for p in range(handle.number_of_pages):
            handle.set_page(p)  
            current_page = handle[:]
            im = Image.fromarray(current_page)
            images.append(im)
    
    return images

# Binning image
def binning(image, bin_factor, size):
    mat = np.array(image)
    mat_reshape = mat.reshape(mat.shape[0]//bin_factor, 2, mat.shape[1]//bin_factor, 2)
    bin_mat = np.mean(mat_reshape, axis=(1, 3))
    
    resize = cv2.resize(bin_mat, (size, size), interpolation=cv2.INTER_CUBIC)
    resize_im = Image.fromarray(resize)
    if resize_im.mode != 'RGB':
        resize_im = resize_im.convert("RGB")

    return resize_im

def crop(image, x, y, h, w):
    image = np.array(image)
    crop_im = image[y:y+h, x:x+w]
    crop_im = Image.fromarray(np.uint8(crop_im))

    return crop_im

def write_data(output_path, images):
    with pytiff.Tiff(output_path, "w") as handle:
        for i in range(len(images)):
            data = images[i]
            handle.write(data, method="tile")


if __name__ == "__main__":
    path1 = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM_pristine_segmentation.tif"
    path2 = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_pristine_raw.tif"
    path3 = "/Users/haoruilong/Dataset for Battery/Pristine/XTM_pristine_raw.tif"

    images_2 = load_data(path2)
    images_3 = load_data(path3)
    
    for i in range(len(images_2)):
        im = images_2[i]
        im = Image.fromarray(np.uint8(im))
        cropped = crop(im, 200, 200, 512, 512)
        binned = special_binning(cropped, 2)
        
        #cropped.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_raw/original/sample_{i}.jpeg", "JPEG")
        #binned.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_raw/sample_{i}.jpeg", "jpeg")

    for i in range(len(images_3)):
        im = images_3[i]
        im = Image.fromarray(np.uint8(im))
        cropped = crop(im, 200, 200, 512, 512)
        binned = special_binning(cropped, 2)

        #cropped.save(f"/Users/haoruilong/Dataset for Battery/Pristine/XTM_raw/original/sample_{i}.jpeg", "JPEG")
        #binned.save(f"/Users/haoruilong/Dataset for Battery/Pristine/XTM_raw/sample_{i}.jpeg", "jpeg")
    
    #outpath_2 = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_raw/PTY_raw_512.tif"
    #out_im = [cv2.imread(file) for file in glob.glob("/Users/haoruilong/Dataset for Battery/Pristine/PTY_raw/bin_2/*.jpeg")]

    #write_data(outpath_2, out_im)
