'''
This file implements ORB feature detection using OpenCV
ORB uses:
1. FAST for detecting candidate corner points (noisy)
2. BRIEF (descriptor) gives binary map of the points
3. ORB (rotator) rotates the descriptor thus making it rotation invariant

Note: Even though ORB is feature detection algorithm, it prioritizes detecting those features 
whose descriptors can be matched across multiple images even when they're rotated (rotational invariance)
'''
# Importing necessary libraries
import numpy as np
import cv2

def compute_ORB(in_img: np.ndarray):
    '''
    This function implements ORB feature detection for practical CV pipelines
    Args:
        in_img: Input image array
    
    Returns:
        (key_points, desc): Tuple of key points (corner coordinates) and descriptor 
    '''
    orb = cv2.ORB_create()
    key_points, desc = orb.detectAndCompute(in_img, None) # 2nd arg mask is None
    x, y = [], []

    for points in key_points:
        xs, ys = points.pt
        x.append(xs)
        y.append(ys)
    
    return x,y, desc