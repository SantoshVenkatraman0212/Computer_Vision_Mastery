'''
This file handles the OpenCV implementation of Shi-Tomasi Feature Detection
'''
# Importing necessary libraries
import numpy as np
import cv2

def shi_tomasi(in_img: np.ndarray):
    '''
    This function implements OpenCV's Shi-Tomasi feature detection

    Args:
        in_img: Input image array
    
    Returns:
        shi_tomasi_corners: int32 array of corner (feature) points
    '''
    # maxCorners -> No of feature points, qualityLevel -> Threshold, minDistance -> Neighborhood size
    corners = cv2.goodFeaturesToTrack(in_img, maxCorners = 200, qualityLevel = 0.01, minDistance = 10)
    corners = np.int32(corners)

    return corners