import numpy as np
from PIL import Image
import cv2
from Preprocess import crop

# Load the original image
img = Image.open("crop.jpeg")

# Convert the image to a NumPy array
img_arr = np.array(img)

# Get the height and width of the original image
height, width = img_arr.shape[:2]

# Define the bin factor
bin_factor = 2

# Calculate the new height and width of the binned image
new_height = height // bin_factor
new_width = width // bin_factor

# Define the kernel size and stride
kernel_size = (bin_factor, bin_factor)
stride = (1, 1)

# Use np.mean to bin the image
binned_img = np.mean(img_arr[:new_height*bin_factor:stride[0], :new_width*bin_factor:stride[1]].reshape(new_height, bin_factor, new_width, bin_factor), axis=(1,3))

binned_img = cv2.resize(binned_img, (512, 512), interpolation=cv2.INTER_NEAREST)

# Convert the binned image back to a PIL Image object
binned_img = Image.fromarray(np.uint8(binned_img))

# Show the binned image
binned_img.save('Bin_2_512.jpeg')

def downsampling(image, scale, sigma):
    im = Image.open(image)
    im_arr = np.array(im)
    
    im_arr_blur = cv2.GaussianBlur(im_arr, (3, 3), sigma)
    im_arr_down = cv2.resize(im_arr_blur, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_NEAREST)

    new_im = Image.fromarray(np.uint8(im_arr_down))
    return new_im

def special_binning(image, bin_factor):
    image = Image.open(image)
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

if __name__ == "__main__":
    downx2_special_binx2 = downsampling('special_binx2.jpeg', 2, 0)
    downx4_special_binx2 = downsampling('special_binx2.jpeg', 4, 0)
    downx8_special_binx2 = downsampling('special_binx2.jpeg', 8, 0)

    downx2_special_binx2.save('downx2.jpeg')
    downx4_special_binx2.save('downx4.jpeg')
    downx8_special_binx2.save('downx8.jpeg')