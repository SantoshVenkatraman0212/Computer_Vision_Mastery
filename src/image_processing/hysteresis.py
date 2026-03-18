'''
This file performs hysteresis on the double threshold array to obtain the true edges of all
objects depicted in the image and effectively removing all noise
'''
# Importing necessary libraries
import numpy as np

# def compute_hysteresis(double_thresh: np.ndarray):
#     '''
#     This function performs hysteresis operation on the input double threshold image array.
#     All strong edges and all weak edges connected to strong edges will be retained as they are
#     the true edges of objects in the image. An isolated weak edge is noise.

#     Args:
#         double_thresh: input double threshold array

#     Returns:
#         noiseless_img: image array with the true edges devoid of all noise
#     '''
#     H, W = double_thresh.shape
#     noiseless_img = np.zeros(H, W, dtype = double_thresh.dtype)

#     for i in range(H):
#         for j in range(W):
#             if double_thresh[i][j] == 75:
#                 # Prev left
#                 if (j - 1) >= 0:
#                     if double_thresh[i - 1][j] == 255:
#                         noiseless_img[i][j] = double_thresh[i][j]
#                         continue
#                 else:
#                     if (i - 1) >= 0:

#             else:
#                 continue

    