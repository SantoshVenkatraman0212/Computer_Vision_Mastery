'''
This file performs hysteresis on the double threshold array to obtain the true edges of all
objects depicted in the image and effectively removing all noise
'''
# Importing necessary libraries
import numpy as np

def compute_hysteresis(double_thresh: np.ndarray):
    '''
    This function performs hysteresis operation on the input double threshold image array.
    All strong edges and all weak edges connected to strong edges will be retained as they are
    the true edges of objects in the image. An isolated weak edge is noise.

    Args:
        double_thresh: input double threshold array

    Returns:
        noiseless_img: image array with the true edges devoid of all noise
    '''
    H, W = double_thresh.shape
    '''Making a copy of the double threshold matrix as we need updated pixel values at every 
    stage'''
    noiseless_img = double_thresh.copy()

    # Making a list of combinations of all 8 neighbors for each pixel
    neighbors_list = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    # Getting the coordinates of only the strong pixels
    strong_pix_list = []
    for i in range(H):
        for j in range(W):
            if double_thresh[i][j] == 255:
                strong_pix_list.append((i, j))
    
    while len(strong_pix_list) != 0:
        '''As long as the stack isn't empty we check each neighbor, make them into strong pixels
           and also add each neighbor to the stack. This way, every pixel is checked in a chain
        '''
        s_i, s_j = strong_pix_list.pop()
        for n in neighbors_list:
            # Getting neighbor coordinates
            n_i = s_i + n[0]
            n_j = s_j + n[1]
            if n_i >= 0 and n_i < H and n_j >= 0 and n_j < W:
                # Checking if the neighbor coordinates are valid (Reject Out of Bounds)
                if noiseless_img[n_i][n_j] == 75:
                    noiseless_img[n_i][n_j] = 255
                    strong_pix_list.append((n_i, n_j))
    
    # Removing the residual noise
    # only 0 and 255 should remain and isolated 75 should be 0
    noiseless_img[noiseless_img == 75] = 0
    
    return noiseless_img


    