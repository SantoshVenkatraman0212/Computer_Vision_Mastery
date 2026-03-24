'''
This file implements FAST corner detection using OpenCV
FAST is very different from Harris and Shi-Tomasi in the sense that it doesn't use gradients and convolution at all
FAST just checks if a certain pixel's intensity (value) changes drastically i.e. much higher or lower than the threshold compared to the neighbors 
'''
# Importing necessary libraries
import numpy as np
import cv2

def compute_FAST(in_img: np.ndarray):
    '''
    This function implements OpenCV's FAST corner detection
    Args:
        in_img: Input image array
    
    Returns:
        fast_coords: A tuple of list of x and y coordinates to overlay on the original image for corner detection
    '''
    # Creating an instance for FAST
    fast = cv2.FastFeatureDetector_create(threshold = 20, nonmaxSuppression = True)
    # Detecting the corner points
    fast_points = fast.detect(in_img, None)
    # Preparing a list of x and y coordinates of the corner points
    x, y = [], []
    for points in fast_points:
        # fast_points are as OpenCV keypoint objects
        # Therefore, for each point we have to use .pt to get a tuple of x and y coordinates
        xs, ys = points.pt
        x.append(xs)
        y.append(ys)
    
    return x, y
