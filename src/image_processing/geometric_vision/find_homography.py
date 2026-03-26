'''
This file computes homography matrix using OpenCV
'''
# Importing necessary libraries
import numpy as np
from .src_dest_mapping import get_src_dest_matches
import cv2

def compute_homography(src_pts, dest_pts):
    '''
    This function uses OpenCV to compute homography matrix and mask
    from source and destination image key points
    '''
    # RANSAC filters out only the best matches
    H, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC)

    return H, mask