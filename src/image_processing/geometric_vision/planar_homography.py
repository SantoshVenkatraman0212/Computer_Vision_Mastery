'''
This file implements the entire homography pipeline:
1. Obtains key points and matches using ORB
2. Gets the source and destination points from the matches
3. Computes homography matrix and mask
4. Warps the source image and returns it
'''
# Importing necessary libraries
import numpy as np
from .src_dest_mapping import get_src_dest_matches
from .find_homography import compute_homography
from .warping import get_warped_img

def compute_planar_homography(src_img: np.ndarray, dest_img: np.ndarray):
    # Getting source and destination points
    src_pts, dest_pts = get_src_dest_matches(src_img, dest_img)
    # Computing homography matrix and mask
    H, _ = compute_homography(src_pts, dest_pts)
    # Warping the source image
    warped_img = get_warped_img(src_img, H, dest_img)

    return warped_img