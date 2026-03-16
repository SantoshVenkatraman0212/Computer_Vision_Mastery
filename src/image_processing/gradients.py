'''
This file is for computing 1st order and 2nd order derivatives
This is the flow:
Image matrix -> Gaussian Blur (Denoising) -> Sobel-X, Sobel-Y (First order derivative) -> sqrt(Gx^2 + Gy^2) Gives total edges -> arctan(Gx, Gy) Gives edge orientation 
'''
# Importing necessary libraries
import numpy as np
from .convolution import convolve
from .filters import kernel_dict
import math

# Function for gaussian blur
def gaussian_blur(in_img: np.ndarray):
    '''
    This function performs convolution operation on the input image
    using gaussian blur as the filter

    Args:
        in_img: Input image as a numpy array

    Returns:
        smooth_img: Image denoised and smoothened using gaussian blur kernel
    '''
    smooth_img = convolve(in_img, 'gaussian_blur', 3, 1, 'valid')
    return smooth_img

def sobel_gradients(smooth_img: np.ndarray):
    '''
    This function computes Sobel-X (Gx -> Vertical gradient, Gy -> Horizontal gradient)
    Args:
        smooth_img: Accepts smoothened image array
    
    Returns:
        (Gx: Sobel-X array, Gy: Sobel-Y array)
    '''
    Gx = convolve(smooth_img, 'sobel_x', 3, 1, 'valid') 
    Gy = convolve(smooth_img, 'sobel_y', 3, 1, 'valid')

    return Gx, Gy

def all_edges(Gx: np.ndarray, Gy: np.ndarray):
    '''
    This function computes all the edges in the image upon applying sobel filters
    Args:
        Gx: Sobel-X array
        Gy: Sobel-Y array
    
    Returns:
        magnitude: magnitude of the combined derivatives
    '''
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    return magnitude

def orientation(Gx: np.ndarray, Gy: np.ndarray):
    '''
    This function computes the angle of orientation of the edges
    Args:
        Gx: Sobel-X array
        Gy: Sobel-Y array
    
    Returns:
        Theta: angle in degrees
        90 degrees: horizontal edge
        0 degrees: vertical edge
        45 degrees: diagnoal edge
    '''
    theta = np.arctan2(Gx, Gy)

    return theta