import numpy as np
from PIL import Image
import cv2

def downsampling(image, scale, sigma):
    im = Image.open(image)
    im_arr = np.array(im)
    
    im_arr_blur = cv2.GaussianBlur(im_arr, (3, 3), sigma)
    im_arr_down = cv2.resize(im_arr_blur, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_NEAREST)

    new_im = Image.fromarray(np.uint8(im_arr_down))
    return new_im

def special_binning(image, bin_factor):
    #image = Image.open(image)
    im = np.array(image)
    h, w = im.shape[:2]


    new_height = h // bin_factor
    new_width = w // bin_factor

    im_reshape = im[:new_height*bin_factor, :new_width*bin_factor].reshape(new_height, bin_factor, new_width, bin_factor)
    bin_im = np.mean(im_reshape, axis=(1, 3))
    
    bin_im_blur = cv2.GaussianBlur(bin_im, (3, 3), 0)
    bin_im_resize = cv2.resize(bin_im_blur, (h, w), interpolation=cv2.INTER_NEAREST)
    bin_im_resize = Image.fromarray(np.uint8(bin_im_resize))
    
    return bin_im_resize
