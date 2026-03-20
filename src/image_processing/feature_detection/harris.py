'''
This file implements the entire harris feature detection pipeline right from gaussian blurring 
to response, NMS and refinement
'''
# Importing necessary libraries
import numpy as np
from .harris_response import compute_harris_response
from .structure_tensor import compute_structure_tensor

def compute_harris(response_array: np.ndarray):
    '''
    This function creates a threshold for corners, creates mask indices for the harris 
    repsonse array, then performs Harris NMS on R. Note: This NMS is different from Canny Edge
    NMS.
    '''
    # Making a zeros array with response_array dimensions
    H, W = response_array.shape
    arr = np.zeros((H, W), dtype = response_array.dtype)
    threshold_ratio = 0.01 # (recommended harris threshold value)
    threshold = threshold_ratio * response_array.max()
    # The only indices in the response response to suppress or allow
    thresh_ind = []
    for a in range(H):
        for b in range(W):
            if response_array[a][b] >= threshold:
                thresh_ind.append((a, b))

    # Using a 3 x 3 grid of neighbors as this is the most tested and robust value
    # A 3 x 3 grid has 8 neighbors and one center pixel
    neighbors_list = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    # Setting pixel values for only the ones above or equal to the threshold
    for r_indices in thresh_ind:
        k = True
        # Iterating through the neighbors list
        for n in neighbors_list:
            i = r_indices[0] + n[0]
            j = r_indices[1] + n[1]
            # Checking if the index is valid
            if i >= 0 and i < H and j >= 0 and j < W:
                if response_array[r_indices[0]][r_indices[1]] > response_array[i][j]:
                    continue
                else:
                    k = False
                    break
        if k is True:
            arr[r_indices[0]][r_indices[1]] = 1
    
    return arr


