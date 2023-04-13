import numpy as np
from PIL import Image

# Load image and convert to grayscale
image = Image.open("/Users/haoruilong/Documents/Livepaper/view.jpeg").convert("L")

# Convert image to NumPy array
image_array = np.array(image)

# Define binning factor (e.g. 2 for halving the image size)
bin_factor = 2

# Binning (downsampling) the image using NumPy mean function
binned_array = np.mean(image_array.reshape(image_array.shape[0]//bin_factor, bin_factor, 
                                            image_array.shape[1]//bin_factor, bin_factor), axis=(1,3))

# Convert the binned array back to image
binned_image = Image.fromarray(np.uint8(binned_array))

# Display the original and binned images
image.show()
binned_image.show()

# Print the variables