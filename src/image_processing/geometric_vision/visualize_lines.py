'''
This file provides a function that can be used for plotting points and matching lines for epipolar geometry
'''
# Importing necessary libraries
import cv2
import numpy as np

def draw_epilines(img, lines, pts):
    '''
    This function plots lines on the second image that 
    correspond to the points in the first image  
    Args:
        img: Image-B
        lines: Matching lines in 3-D coordinate system
        pts: Matching points
    
    Returns:
        final_img: Image-B with matching points and lines
    '''
    # Converting the grayscale image to BGR (colour) for coloured pts 
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Getting the height and width of the image
    h, w = img.shape[: 2]
    # Iterating through each 3-D line(plane) pts
    for r, pt in zip(lines, pts):
        # Getting the x, y and z coords of the plane
        a, b, c = r[0]
        # Converting to 2-D
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-((a * w) + c)/ c)
        # Plotting the line
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
        # Plotting the matching points
        cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
    
    return img
