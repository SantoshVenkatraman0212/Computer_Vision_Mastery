'''
This file demonstrates and plots ORB feature points and descriptors for an explainable and usable CV pipeline
Note: key points will be plotted here, while descriptor will be shown in the feature matching demo
'''
# Importing necessary libraries
import numpy as np
from config.paths import orb_demo_img_path, sample_img_path
import os
import cv2
from src.image_processing.feature_detection.ORB import compute_ORB
import matplotlib.pyplot as plt

# Loading in the sample image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# Computing descriptor and key points
kp_x, kp_y, desc = compute_ORB(sample_img)
# Plotting the ORB feature points
plt.figure(figsize = (10, 7))
plt.imshow(sample_img, cmap = 'gray')
plt.scatter(kp_x, kp_y, s = 5, c = 'red')
plt.title('ORB feature detection')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{orb_demo_img_path}/ORB_demo.png', dpi = 300, bbox_inches = 'tight')