import numpy as np
from PIL import Image
import cv2

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

binned_img = cv2.resize(binned_img, (1000, 1000), interpolation=cv2.INTER_NEAREST)

# Convert the binned image back to a PIL Image object
binned_img = Image.fromarray(np.uint8(binned_img))

# Show the binned image
binned_img.save('Sbin_2.jpeg')
