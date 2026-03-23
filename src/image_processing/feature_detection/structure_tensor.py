'''
This file handles the initial setup steps required for harris corner / feature detection
'''
# Importing necessary libraries
import numpy as np
from src.image_processing.edge_detection.gradients import gaussian_blur, sobel_gradients, all_edges

def compute_structure_tensor(in_img: np.ndarray):
    '''
    This function does the initial setup image processing steps and builds 
     structure tensors leading up to harris feature
    detection
    1. Input image smoothening (Gaussian Blur)
    2. Gradients (Sobel-X, Sobel-Y)
    3. Computing Ixx, Iyy, and Ixy

    Args:
        in_img: Input image array
    
    Returns:
        struct_tensor: Tuple of (Ixx, Iyy, and Ixy) in this order
    '''
    # Smooth Image
    smooth_img = gaussian_blur(in_img)
    # Sobel gradients
    Ix, Iy = sobel_gradients(smooth_img)
    # Structure tensor computation
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    # Smoothening the structure tensors
    Ixx, Iyy, Ixy = gaussian_blur(Ixx), gaussian_blur(Iyy), gaussian_blur(Ixy)

    return Ixx, Iyy, Ixy
