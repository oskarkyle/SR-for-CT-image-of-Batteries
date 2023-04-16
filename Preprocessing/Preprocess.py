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

def write_data(output_path, images):
    with pytiff.Tiff(output_path, "w") as handle:
        for i in range(len(images)):
            data = np.array(images[i], np.uint8)
            handle.write(data, method="tile")


if __name__ == "__main__":
    path = "/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM_pristine_segmentation.tif"
    images = []
    bin_imgs = []
    
    images = load_data(path)
    
    for i in range(len(images)):
        im = images[i]
        crop_im1 = crop(im, 0, 0, 1000, 1000)
        crop_im2 = crop(im, 0, 1000, 1000, 1000)
        crop_im3 = crop(im, 1000, 0, 1000, 1000)
        crop_im4 = crop(im, 1000, 1000, 1000, 1000)

        bin_im1 = binning(crop_im1, 2, 1000)
        bin_im2 = binning(crop_im2, 2, 1000)
        bin_im3 = binning(crop_im3, 2, 1000)
        bin_im4 = binning(crop_im4, 2, 1000)

        bin_img = [bin_im1, bin_im2, bin_im3, bin_im4]
        bin_imgs.append(bin_img)

        bin_im1.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM/Bin_2/sample_{i}_1.jpeg", "JPEG")
        bin_im2.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM/Bin_2/sample_{i}_2.jpeg", "JPEG")
        bin_im3.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM/Bin_2/sample_{i}_3.jpeg", "JPEG")
        bin_im4.save(f"/Users/haoruilong/Dataset for Battery/Pristine/PTY_XTM/Bin_2/sample_{i}_4.jpeg", "JPEG")