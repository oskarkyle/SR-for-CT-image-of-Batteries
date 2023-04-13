import pytiff
import numpy as np
from PIL import Image
import cv2

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
    crop_im = Image.fromarray(crop_im)

    return crop_im

if __name__ == "__main__":
    path = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM_pristine_segmentation.tif"
    images = []
    images = load_data(path)
    test = images[0]

    crop_im = crop(test, 0, 0, 1000, 1000)
    crop_im.save('crop.jpeg')
    bin = binning(crop_im, 2, 1000)
    bin.save('bin_2.jpeg')