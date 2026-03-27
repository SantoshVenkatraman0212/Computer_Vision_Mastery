'''
This file deals with image stitching i.e. warping and merging 
2 images to create a proper panorama like effect
'''
# Importing necessary libraries
import numpy as np
from .src_dest_mapping import get_src_dest_matches
from .find_homography import compute_homography
import cv2

def img_stitcher(img_a: np.ndarray, img_b: np.ndarray):
    '''
    This function stitches an image with its shifted version for panoramic effect
    It does the following:
    1. Computes homography
    2. Make an image canvas with w = w1 + w2 and h = max(h1, h2)
    3. Warp image 1 to fit in the canvas
    4. Place image 2 in the remaining space for stitching

    Args:
        img_a: Original image array
        img_b: Shifted image array
    
    Returns:
        stitched_img: Stitched image by warping and merging img_a and img_b
    '''
    # Getting Source and destination pts with ORB and matches
    src_pts, dest_pts = get_src_dest_matches(img_a, img_b)
    # Getting homography matrix and mask
    H, mask = compute_homography(src_pts, dest_pts)
    # Making canvas to fit both the images
    h1, w1 = img_a.shape
    h2, w2 = img_b.shape
    c_w = w1 + w2
    c_h = max(h1, h2)
    # Initializing the canvas 
    canvas = np.zeros((c_h, c_w), dtype = np.uint8)
    # Warp img_a
    # Now img_a has been warped in such a way that it stretches across the entire canvas
    warped_img_a = cv2.warpPerspective(img_a, H, (c_w, c_h)) # Here it's width, height as OpenCV uses (width, height)
    # Now we're placing img_b which is left cropped image_a on the top left of the canvas
    # Place img_b in the canvas
    canvas[: h2, : w2] = img_b
    # Masks for non-zero pixel values
    mask_a = warped_img_a > 0
    mask_b = canvas > 0
    # Creating a stitched_img array which is the same size as that of the canvas
    # Creating a full pitch black array with the size as that of the canvas
    stitched_img = np.zeros_like(canvas)
    # The idea here is, as img_a has been stretched, and img_b is there in the canvas
    # We're finding and placing only those pixels in warped img_a, that complete the remaining image rather than overlapping with img_b
    # Non zero pixel values in warped_img_a which aren't there in img_b
    stitched_img[mask_a & ~mask_b] = warped_img_a[mask_a & ~mask_b]
    # Non zero pixel values in warped_img_b which aren't there in img_a
    stitched_img[~mask_a & mask_b] = canvas[~mask_a & mask_b]
    # When there's an overlap, we take an average for smooth blending
    stitched_img[mask_a & mask_b] = (((warped_img_a[mask_a & mask_b]).astype(np.float32) + (canvas[mask_a & mask_b]).astype(np.float32)) / 2).astype(np.uint8)

    return stitched_img



