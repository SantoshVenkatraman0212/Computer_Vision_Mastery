'''
This file demonstrates and plots FAST corner points on the input image
'''
# Importing necessary libraries
import numpy as np
import cv2
import os
from config.paths import fast_demo_img_path, sample_img_path
from src.image_processing.feature_detection.FAST import compute_FAST
import matplotlib.pyplot as plt

# Loading in the image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# Getting the FAST corner points
fast_x, fast_y = compute_FAST(sample_img)

# Plotting the points
plt.figure(figsize = (10, 7))
plt.imshow(sample_img, cmap = 'gray')
plt.scatter(fast_x, fast_y, s = 5, c = 'red')
plt.title('FAST corner detection')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{fast_demo_img_path}/FAST_demo.png', dpi = 300, bbox_inches = 'tight')