'''
This file computes harris response using the computation on structure tensor outputs
'''
# Importing necessary libraries
import numpy as np
from .structure_tensor import compute_structure_tensor

def compute_harris_response(Ixx: np.ndarray, Iyy: np.ndarray, Ixy: np.ndarray):
    '''
    This function computes determinant, trace and uses these quantities to get the 
    harris repsonse

    det = Ixx*Iyy - Ixy^2
    trace = Ixx + Iyy
    response = det - k * trace^2

    Args:
        Ixx: square of Input x (Sobel-X) structure tensor 
        Iyy: square of Input y (Sobel-Y) structure tensor 
        Ixy: element-wise product of Ix and Iy
    '''
    k = 0.04 # (Most recommended scale value)
    # Determinant
    det = Ixx * Iyy - Ixy ** 2
    # Trace
    trace = Ixx + Iyy
    # Harris response map
    response = det - (k * (trace ** 2))
    
    return response