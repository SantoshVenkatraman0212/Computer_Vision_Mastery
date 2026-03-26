'''
This file demonstrates planar homography i.e. how a source image is transformed
to give the destination image
'''
# Importing necessary libraries
from config.paths import sample_img_path, homography_demo_img_path
from src.image_processing.geometric_vision.planar_homography import compute_planar_homography
from src.image_processing.feature_detection.ORB_feature_matching import rotate_image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Loading in the image
source_image = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)
# Rotating the image
destination_image = rotate_image(source_image, 30)

# Getting the transformed source image
warped_image = compute_planar_homography(source_image, destination_image)


# Plotting the image
def img_plotter(img: np.ndarray, title: str):
    plt.imshow(img, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

plt.figure(figsize = (10, 7))
plt.subplot(2, 3, 1)
# Source image
img_plotter(source_image, 'Source image')
# Destination image
plt.subplot(2, 3, 2)
img_plotter(destination_image, 'Destination image')
# Transformed image
plt.subplot(2, 3, 3)
img_plotter(warped_image, 'Transformed source image')

plt.savefig(f'{homography_demo_img_path}/homography_demo.png', dpi = 300, bbox_inches = 'tight')


