'''
This file provides the full implementation of canny edge detection pipeline involving:
1. Gaussian Blur (Smoothening out noise and high frequency gradients)
2. Sobel-X, Sobel-Y (Gradients along vertical and horizontal direction)
3. Gradient Magnitude (All gradients across the image)
4. Theta (Gradient Orientation)
5. Non Maximal Suppression (NMS) Retaining only the pixels that have local maxima in 
that direction
6. Double Thresholding (Classifies the pixels as strong, weak and no edges)
7. Hysteresis (Retaining only the true edges) 
'''
# Importing nececssary libraries
import numpy as np
from src.image_processing.edge_detection.gradients import (gaussian_blur, sobel_gradients, all_edges, 
                                            orientation)
from src.image_processing.edge_detection.edge_detection import radians_to_angle_conv, non_maximal_suppresion
from src.image_processing.edge_detection.double_thresholding import double_threshold
from src.image_processing.edge_detection.hysteresis import compute_hysteresis
import cv2
import os
import matplotlib.pyplot as plt

def canny_edge_detector(in_img: np.ndarray):
    '''
    This function performs the full canny edge detection right from gaussian blurring to 
    hysteresis computation

    Args:
        in_img: Input image array
    
    Returns:
        canny_edge_img: Image after canny edge detection
    '''
    # Smoothened image
    smooth_img = gaussian_blur(in_img)
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

    # Computing the hysteresis matrix
    canny_edge_img = compute_hysteresis(double_threshold_matrix)

    return canny_edge_img
