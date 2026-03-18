'''
This code demonstrates double thresholding on Non Maximal Suppression Array
'''
# Importing necessary libraries
import numpy as np
from config.paths import sample_img_path, double_threshold_img_path
from src.image_processing.gradients import (gaussian_blur, sobel_gradients, all_edges, 
                                            orientation)
from src.image_processing.edge_detection import radians_to_angle_conv, non_maximal_suppresion
from src.image_processing.double_thresholding import double_threshold
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

# Converting theta from radians to +ve angles
theta_angle = radians_to_angle_conv(theta) 

# Computing NMS
nms = non_maximal_suppresion(magnitude, theta_angle)
# Normalizing for consistency of pixel values
norm_nms = (nms / nms.max()) * 255
# Computing well-defined edge matrix using double thresholding
double_threshold_matrix = double_threshold(nms)
# Plotting the double threshold matrix and compare with NMS matrix
plt.figure(figsize = (10, 7))
# NMS matrix
plt.subplot(1, 2, 1)
plt.imshow(nms, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('Image after Non maximal suppression')
plt.axis('off')
plt.tight_layout()
# Double threshold matrix
plt.subplot(1, 2, 2)
plt.imshow(double_threshold_matrix, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('Sample image after double thresholding')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{double_threshold_img_path}/nms_vs_double_thresh.png', dpi = 300, 
            bbox_inches = 'tight')