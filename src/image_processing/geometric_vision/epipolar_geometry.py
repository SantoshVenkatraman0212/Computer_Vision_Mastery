'''
This file implements epipolar geometry where lines for matching points are drawn in image-B
with reference to image-A
'''
# Importing necessary libraries
import cv2
import numpy as np
from .visualize_lines import draw_epilines
from .src_dest_mapping import get_src_dest_matches

def compute_epipolar(src_img: np.ndarray, dest_img: np.ndarray):
    '''
    This function computes the final image with the epipolar lines i.e.
    the lines that represent the matching points in image-B
    It does the following:
    1. Computes source and destination points using ORB
    2. Computes mask and fundamental matrix that represents the relationship
    3. Computes epipolar lines
    4. Plots the matching points and lines on image-B
    '''
    # Getting the source and the destination points
    src_pts, dest_pts = get_src_dest_matches(src_img, dest_img)
    # Computing the mask and fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dest_pts, cv2.FM_RANSAC)
    # Keeping only the best points
    src_pts, dest_pts = src_pts[mask.ravel() == 1], dest_pts[mask.ravel() == 1]
    # Now plotting the epipolar lines
    # 1 -> image-A
    # F -> fundamental matrix
    lines = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F)

    final_img = draw_epilines(dest_img, lines, dest_pts)

    return final_img

