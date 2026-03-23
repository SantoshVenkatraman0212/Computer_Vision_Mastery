'''
This file plots and compares the capability of the manually built canny edge pipeline with
Open-CV's canny edge implementation
'''
# Importing necessary libraries
import numpy as np
import cv2
from config.paths import sample_img_path, canny_edge_comparison_img_path
from src.image_processing.edge_detection.canny_edge_detection import canny_edge_detector
import os
import matplotlib.pyplot as plt

# Original Image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}')

# Manually computed canny edge image
manual_canny_edge_image = canny_edge_detector(sample_img)
# OpenCV's canny edge implementation 
# Since, CV2 works on absolute threshold, 3 ranges will be chosen
# Top 15% and above and 5% - 15%
# Recommended Ranges for OpenCV Canny edge thresholds: 30 - 100; 50 - 150
opencv_canny_1 = cv2.Canny(sample_img, 255 * 0.05, 255 * 0.15)
opencv_canny_2 = cv2.Canny(sample_img, 30, 100)
opencv_canny_3 = cv2.Canny(sample_img, 50, 150)

def img_plotter(in_img: np.ndarray, title: str):
    '''
    This function plots the input image array in gray scale with the desired title
    Args:
        in_img: Input image array
        title: Title of the plot
    '''
    plt.imshow(in_img, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

# Plotting all 4 images in a subplot
plt.figure(figsize = (10, 7))
# Manual Canny Edge Detection
plt.subplot(2, 2, 1)
img_plotter(manual_canny_edge_image, 'Manual Cannny Edge implementation')    
# OpenCV Canny Edge with low:5%, high: 15%
plt.subplot(2, 2, 2)
img_plotter(opencv_canny_1, 'OpenCV Canny Edge with low:5%, high: 15%')
# OpenCV Canny Edge with low:30, high: 100
plt.subplot(2, 2, 3)
img_plotter(opencv_canny_2, 'OpenCV Canny Edge with low:30, high: 100')
# OpenCV Canny Edge with low:50, high: 150
plt.subplot(2, 2, 4)
img_plotter(opencv_canny_3, 'OpenCV Canny Edge with low:50, high: 150')

# Saving the img
plt.savefig(f'{canny_edge_comparison_img_path}/canny_edge_comparison_demo.png', dpi = 300, 
            bbox_inches = 'tight')



