'''
This file warps i.e. transforms the source image into destination image
using perspective shift
'''
# Importing necessary libraries
import cv2
import numpy as np

def get_warped_img(src_img: np.ndarray, H: np.ndarray, dest_img: np.ndarray):
    '''
    This function warps the source image using homography matrix, and 
    inverse perspective shift
    Args:
        src_img: Source input image
        H: Homography matrix
        dest_img: rotated source image

    Returns:
        warped_img: transformed source image
    '''
    h, w = dest_img.shape
    warped_img = cv2.warpPerspective(src_img, H, (h, w))

    return warped_img