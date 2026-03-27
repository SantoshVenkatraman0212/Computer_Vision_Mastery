'''
This file demonstrates image stitching to simulate panorama
'''
# Importing necessary libraries
import numpy as np
from src.image_processing.geometric_vision.image_stitching import img_stitcher
from config.paths import sample_img_path, image_stitching_img_path
import os
import cv2
import matplotlib.pyplot as plt

# Loading in the image
img_a = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)
img_b = img_a[: , 100:]

# Getting the stitched panoramic image
stitched_img = img_stitcher(img_a, img_b)

# Plotting the images
def img_plot(in_img: np.ndarray, title: str):
    plt.imshow(in_img, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

plt.figure(figsize = (10, 7))
plt.subplot(2, 2, 1)
img_plot(img_a, 'Image-A')
plt.subplot(2, 2, 2)
img_plot(img_b, 'Left cropped image-A')
plt.subplot(2, 2, 3)
img_plot(stitched_img, 'Stitched images')

plt.savefig(f'{image_stitching_img_path}/image_stitching_demo.png', dpi = 300, bbox_inches = 'tight')