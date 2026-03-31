'''
This file demonstrates epipolar geometry implementation
'''
# Importing necessary libraries
import numpy as np
import cv2
import os
from config.paths import epipolar_geometry_img_path, sample_img_path
from src.image_processing.geometric_vision.epipolar_geometry import compute_epipolar
from src.image_processing.feature_detection.ORB_feature_matching import rotate_image
import matplotlib.pyplot as plt

# Loading in the image
src_image = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)
# Destination image
dest_image = rotate_image(src_image, 60)
# Computing image with epipolar lines
final_img = compute_epipolar(src_image, dest_image)

def img_plotter(img: np.ndarray, title: str):
    '''
    This function is specifically for quick subplots
    Args: 
        img: Input image array
        title: Title of the image
    '''
    plt.imshow(img, cmap = 'binary')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

plt.figure(figsize = (10, 7))
plt.subplot(2, 2, 1)
img_plotter(src_image, 'Source image')
plt.subplot(2, 2, 2)
img_plotter(dest_image, 'Destination image')
plt.subplot(2, 2, 3)
img_plotter(final_img, 'Epipolar image')

plt.savefig(f'{epipolar_geometry_img_path}/epipolar_img.png', dpi = 300, bbox_inches = 'tight')

