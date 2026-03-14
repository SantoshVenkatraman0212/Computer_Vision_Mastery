'''
This file performs convolution operation that is the basis of filters like sobel, laplace etc, and this algorithm
is what CNNs rely on for capturing features
'''
# Importing necessary libraries
import numpy as np
from .filters import kernel_dict

def convolve(in_img, kernel_type: str, kernel_size, stride, padding: str):
    '''
    This function performs convolution operation on input image array
    
    Args:
        
        in_img: Input image array 
        kernel_type: Type of filter we want to apply
        kernel_size: Filter size
        stride: No of steps the kernel moves (horizontal and vertical)
        padding: Whether zero padding is required or not
    
    Returns:

        Returns: Output image array after convolution
    '''
    # Reject unsupported kernel type
    if kernel_type not in kernel_dict:
        raise ValueError('Invalid kernel type')
    # Reject unsupported kernel size
    if kernel_size not in kernel_dict[kernel_type]:
        raise ValueError('Invalid kernel size')

    # With kernel_type we determine filter
    # With the kernel_size we determine dims of the filter
    kernel = kernel_dict[kernel_type][kernel_size]
    # This function only supports same and valid padding
    if padding != 'same' and padding != 'valid':
        raise ValueError('Invalid padding type')
    # Reject negative and 0 stride
    if stride <= 0:
        raise ValueError('Stride has to be positive')
    
    # In case of same padding, the output image after convolution will have the same dimensions as input image
    if padding == 'same':
        p = kernel_size // 2
        # Creating a new zeros array for getting original image array bounded by zero padding
        copy_img = np.zeros(shape = (in_img.shape[0] + (2 * p), in_img.shape[1] + (2 * p)), dtype = in_img.dtype)
        copy_img[p: p + in_img.shape[0], p: p + in_img.shape[1]] = in_img
        in_img = copy_img
    else:
        # If padding is valid, then no modification has to be made to the original image array
        in_img = in_img
    
    # list for overall output image array
    out_img = []
    # list for every row
    row = []
    
    i, j = 0, 0

    while True:
        # element-wise product and sum
        row.append(np.sum(in_img[i: i + kernel.shape[0], j: j + kernel.shape[1]] * kernel))
        # right shift
        j += stride
        # If the kernel goes out of bounds during right shift, the window moves down by stride from left corner
        if j + kernel.shape[1] > in_img.shape[1]:
            out_img.append(row)
            row = []
            # Reinitializing the row list for the next row of elements
            i += stride
            j = 0
            # If the kernel goes out of bounds during vertical shift, then convolution operation is over
            if i + kernel.shape[0] > in_img.shape[0]:
                return np.array(out_img) 



    
    

    
    
        


            
            

    



    

