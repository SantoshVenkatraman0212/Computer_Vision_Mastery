'''
This file performs double thresholding on the Non-Maximal Suppression (NMS) array
'''
# Importing necessary libraries
import numpy as np

def double_threshold(nms_array: np.ndarray):
    '''
    This function creates 2 threshold regions to classify pixels based on their nms values
    value < 5% of max (remove)
    value >= 5% max and value < 15% -> weak edge (make it 75)
    value > 15% max -> strong edge (make it 255)

    Args:
        nms_array: Non maximal suppression array
    
    Returns:
        threshold_array: array depicting well defined edges
    '''
    # Threshold computation
    high, low = nms_array.max() * 0.15, nms_array.max() * 0.05
    
    # Creating a zeroes array for thresholding
    H, W = nms_array.shape
    threshold_array = np.zeros(shape = (H, W), dtype = nms_array.dtype)
    # Creating masks (indices arrays) based on threshold
    # Weak edge mask (5% - 15%)
    weak_edge_mask = (nms_array >= low) & (nms_array < high)
    # Strong edge mask (15% and above)
    strong_edge_mask = nms_array >= high
    threshold_array[weak_edge_mask] = 75
    threshold_array[strong_edge_mask] = 255
    
    return threshold_array