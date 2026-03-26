'''
This file uses ORB to compute, key-points and matches from 2 images, then computes
the best matches for the source and destination image
'''
# Importing necessary libraries
import numpy as np
from src.image_processing.feature_detection.ORB_feature_matching import feature_matching

def get_src_dest_matches(src_img: np.ndarray, dest_img: np.ndarray):
    '''
    This function takes 2 images, computes keypoints and matches using ORB,
    Then computes the best matching key points from the source and destination images 
    Args:
        src_img: input sample image array
        dest_img: rotated src_img i.e. the image we want src_img transformed into
    
    Returns:
        (src_pts, dest_pts): Tuple of source and destination key points
    '''
    # Getting only the keypoints and matches from ORB feature_matching function
    kp_src, _, kp_dest,_, matches = feature_matching(src_img, dest_img)
    # Getting only the keypoints with the strong matches
    src_pts = np.float32([kp_src[match.queryIdx].pt for match in matches])
    dest_pts = np.float32([kp_dest[match.trainIdx].pt for match in matches])

    return src_pts, dest_pts

