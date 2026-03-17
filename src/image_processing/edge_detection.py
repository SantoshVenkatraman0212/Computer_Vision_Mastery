'''
This file performs Non-Maximal Suppression that is essential for getting thin fine, noiseless
edges, that clearly define the boundaries of various objects present in the image
'''
# Importing necessary libraries
import numpy as np
import math
from .gradients import all_edges, orientation

def radians_to_angle_conv(theta: np.ndarray):
    '''
    This function converts the radian values in theta matrix to angles between 0 - 180
    Args:
        theta: input gradient orientation array
    
    Returns:
        theta_angle: theta with angle in degrees
    '''
    theta = (theta * 180) / np.pi
    # Converting -ve angles to positive angles
    theta[theta < 0] += 180

    return theta

def non_maximal_suppresion(magnitude: np.ndarray, theta: np.ndarray):
    '''
    Performs NMS on the input magnitude array based on theta (direction)
    Args:
        magnitude: magnitude array
        theta: direction array
    
    Returns:
        nms: NMS equivalent of magnitude array
    '''
    # No of rows is height; No of cols is width
    H, W = magnitude.shape
    mag_alias = np.zeros(shape = (H, W), dtype = magnitude.dtype)
    # Iterating over the entire magnitude matrix
    # Initializing max pixel
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # We ignore the borders as those edges are problematic and also not reliable
            angle = theta[i][j]

            if (angle >= 0 and angle <= 22.5) or (angle >= 157.5 and angle <= 180):
                # Horizontal
                n1 = magnitude[i][j - 1]
                n2 = magnitude[i][j + 1]
                
            elif angle > 22.5 and angle <= 67.5:
                # Right Diagonal
                n1 = magnitude[i - 1][j + 1]
                n2 = magnitude[i + 1][j - 1]

            elif angle > 67.5 and angle <= 112.5:
                # Vertical
                n1 = magnitude[i - 1][j]
                n2 = magnitude[i + 1][j]

            elif angle > 112.5 and angle <= 157.5:
                # Left Diagonal
                n1 = magnitude[i - 1][j - 1]
                n2 = magnitude[i + 1][j + 1]
                
            if magnitude[i][j] >= n1 and magnitude[i][j] >= n2:
                mag_alias[i][j] = magnitude[i][j]
            
            else:
                continue
    
    return mag_alias
        
            

