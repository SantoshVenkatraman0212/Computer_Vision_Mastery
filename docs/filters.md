# Image Filters
Filters play a crucial role in the field of computer vision, as they can be used for applications such as:

1. edge detection
2. noise removal
3. sharpening
4. smoothing
and so on. 

## Identity Filter
This filter returns the exact same output image as input. Except the center pixel every other value in the kernel is 0

## Blur Filter
A.K.A box filter works on the principle that, each pixel value is the average of its neighbors. So every pixel value is divided by the total no of pixels in the kernel. This reduces the high frequency variations and smoothens noise

## Gaussian Blur Filter
This filter is a weighted average filter in the sense that, the pixels towards the center get higher weights compared to the distant ones. It is designed on gaussian function that produces bell-shaped weightage curve. This filter is heavily used in image preprocessing pipelines in DL image classifiers, and segmentation

## Sobel Filters (x and y)
These are predominant edge detection filters. In sobel-x filter, the kernel has negative pixel values on the left and positive values on the right, thus revealing the vertical edges for objects inside those edges. Similarly sobel-y filter is horizontal edge detector, where the top line is -ve and bottom line is +ve.

## Laplacian Filter
This filter is used for detecting rapid frequency changes in all the directions i.e. the center pixel value varies greatly from all of its 4 neighbors. Center gets -ve and neighbors get strong +ve values.

## Sharpen Filter
This filter increases the sharpness of the images i.e. wherever there are edges or high frequency changes, those will be amplified, thus making the boundaries, textures and surfaces look very sharp.