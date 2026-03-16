'''
This file demonsrtates the following:
1. Image smoothening: Gaussian blur
2. Regions of pixel intensity changes: Sobel X, and Y
3. Magnitude of all the edges
4. Edge orientation
'''
# Importing necessary libraries

import numpy as np
from config.paths import sample_img_path, gradient_demo_img_path
from src.image_processing.gradients import (gaussian_blur, sobel_gradients, all_edges, 
                                            orientation)
import cv2
import os
import matplotlib.pyplot as plt

# Reading in the image
sample_img =  f'{sample_img_path}/{os.listdir(sample_img_path)[0]}'
sample_img = cv2.imread(sample_img, 0)
# Smoothened image
smooth_img = gaussian_blur(sample_img)
# Sobel X and Y
Gx = sobel_gradients(smooth_img)[0]
Gy = sobel_gradients(smooth_img)[1]
# Magnitude
magnitude = all_edges(Gx, Gy)
# Orientation
theta = orientation(Gx, Gy)
# Normalizing for plotting (Can't accurately plot -ve and positive float values)
normalized_theta = (theta + np.pi) / (2 * np.pi)

img_list = [('Original Image', sample_img), ('Smooth Image', smooth_img), 
            ('Vertical Edges', Gx), ('Horizontal Edges', Gy), ('Edge Magnitude', magnitude)]

# Creating a subplot for the images
plt.figure(figsize = (10, 7))
for i, values in enumerate(img_list):

    plt.subplot(3, 3, i + 1)
    plt.imshow(values[1], vmin = 0, vmax = 255, cmap = 'gray')
    plt.title(f'{values[0]}')
    plt.axis('off')

# Since the normalized values are between 0 and 1
plt.imshow(normalized_theta, vmin = 0, vmax = 1, cmap = 'gray')
plt.title('Edge Orientation')
plt.axis('off')

# Saving the figure
plt.savefig(f'{gradient_demo_img_path}/gradient_demo.png', dpi = 300, bbox_inches = 'tight')
