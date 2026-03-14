'''
This file handles filter demos for convolution operation on a real image
'''
# Importing necessary libraries
from src.image_processing.convolution import kernel_dict, convolve
from config.paths import sample_img_path, filtered_sample_img_path
import matplotlib.pyplot as plt
import os
import cv2

# List of filters
kernels = ['identity', 'blur', 'horizontal_edge_detector', 'vertical_edge_detector', 'sharpen', 'gaussian_blur',
           'sobel_x', 'sobel_y', 'laplacian']

# Stitching the name of the sample image with the full path for reading
# Using 0 to load the image in grayscale (monochrome) as the convolve filter works only on single channel
img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)


# Demonstrating a 3 x 3 kernel, with padding set to same for all the filters
plt.figure(figsize = (10, 7))
for index, filters in enumerate(kernels):
    # Getting the convolution result for each filter
    conv_result = convolve(img, filters, 3, 1, 'valid')
    plt.subplot(3, 3, index + 1)
    # Plotting in gray scale, as the filtered image is monochrome
    # Setting vmin, and vmax is crucial to retain the pixel values between 0-255 
    # Matplotlib scales all pixel values by default making them -ve and +ve integers
    plt.imshow(conv_result, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(f'Sample image with \n{filters} kernel')
    plt.axis('off')
    plt.tight_layout()
plt.savefig(f'{filtered_sample_img_path}/sample_img_with_all_kernels_applied.png', dpi = 300, 
                bbox_inches = 'tight')



