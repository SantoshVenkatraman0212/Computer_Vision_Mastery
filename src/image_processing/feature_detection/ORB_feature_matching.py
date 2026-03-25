'''
This file implements feature matching using ORB descriptor.
1. Gets key points and descriptor from original image
2. Rotates the image by 30 degrees and gets keypoints and descriptors
3. Computes matches using hamming distance as the metric
'''
# Importing necessary libraries
import numpy as np
import cv2
from .ORB import compute_ORB

def rotate_image(in_img: np.ndarray, angle: int):
    '''
    This function rotates the input image by the desired angle and returns the rotated image
    Rotation steps:
    1. Computes center of the input image
    2. Computes 2D Rotation matrix using the image center, angle and scale (cropping or no cropping)
    3. Gets the cos, and sine of the angles
    4. Computes new width and height after rotation
    5. Updates the last col of the rotation matrix
    6. Finally image is rotated 
    '''
    H, W = in_img.shape
    # Getting the center of the input image
    center = (H // 2, W // 2)
    # Computing the rotation matrix
    # Scale 1 ensures that the rotated image is of the same aspect ratio
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Getting the cos and sin values from the the rotation matrix
    cos = abs(rotation_matrix[0][0])
    sin = abs(rotation_matrix[0][1])
    # New width = HSin + WCos
    new_w = int((H * sin) +  (W * cos))
    # New height = HCos + WSin
    new_h = int((H * cos) + (W * sin))
    # Updating the 3rd column of rotation matrix with the new center coords
    rotation_matrix[0][2] += (new_w / 2) - center[0]
    rotation_matrix[1][2] += (new_h / 2) - center[1]
    # Rotating the original image
    rotated_img = cv2.warpAffine(in_img, rotation_matrix, (new_w, new_h))

    return rotated_img

def feature_matching(orig_img: np.ndarray, rot_img: np.ndarray, match_type: str = None):
    '''
    This function calls compute_ORB to compute key points and descriptors, 
    computes image feature matches and returns them
    Args:
        orig_img: Original sample image array
        rot_img: Original sample image array rotated through an angle
    
    Returns:
        orig_kp, orig_desc, rot_kp, rot_desc, matches: Tuple of key points and descriptors for 
        original and rotated image and feature matches
    '''
    # Computing key points and descriptors for original and rotated image
    orig_kp, orig_desc = compute_ORB(orig_img)
    rot_kp, rot_desc = compute_ORB(rot_img)
    # Either uses default matching or knn based matching with k specific near matches
    if match_type == 'knn':
        # Creating an instance for feature matcher
        # Normalized Hamming distance is used to determine similarity between the binary image deascriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
        matches = matcher.knnMatch(orig_desc, rot_desc, k = 2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = matcher.match(orig_desc, rot_desc)

    return orig_kp, orig_desc, rot_kp, rot_desc, matches
    
    
    
    


