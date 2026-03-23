'''
This file compares and contrasts harris feature detection implemented 
from scratch to OpenCV's Harris implementation 
'''
# Importing necessary libraries
import numpy as np
from config.paths import sample_img_path, harris_comparison_demo_img_path
from src.image_processing.feature_detection.harris import compute_harris, compute_harris_response, compute_structure_tensor
import cv2
import os
import matplotlib.pyplot as plt

# Loading in the image file
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# ---------- Harris Feature Detection from scratch ----------
# Getting Structure Tensors
Ixx, Iyy, Ixy = compute_structure_tensor(sample_img)
# Getting harris response (manual)
harris_response = compute_harris_response(Ixx, Iyy, Ixy)
# Getting the harris features from the sample image
manual_harris_img = compute_harris(harris_response)
# Normalizing harris points
if manual_harris_img.dtype != np.uint8:
    manual_harris_img = (manual_harris_img - manual_harris_img.min()) / (manual_harris_img.max() - manual_harris_img.min())
# Harris Overlay points
m_x, m_y = np.where(manual_harris_img > 0)

# ---------- Harris Feature Detection by OpenCV ----------
# Converting the gray scale image to float32 to be OpenCV compatible
float_sample_img = np.float32(sample_img)
opencv_harris_response = cv2.cornerHarris(float_sample_img, blockSize = 2, ksize = 3, k = 0.04)
# Getting harris features from opencv harris response
opencv_harris_img = compute_harris(opencv_harris_response)
# Normalizing response
norm_opencv_harris_img = (opencv_harris_img - opencv_harris_img.min()) / (opencv_harris_img.max() - opencv_harris_img.min())
# OpenCV Harris Overlay points
o_x, o_y = np.where(norm_opencv_harris_img > 0)

# Comparing the total number of feature points 
print(f'Total feature points for Harris from scratch: {len(m_x)}')
print(f'Total feature points for OpenCV\'s Harris: {len(o_x)}')

# Plotting and comparing the 2
plt.figure(figsize = (10, 7))
# Harris from scratch
plt.subplot(2, 2, 1)
plt.imshow(sample_img, cmap = 'gray')
plt.title('Harris Feature Detection from scratch')
plt.axis('off')
plt.scatter(m_x, m_y, s = 5)
plt.tight_layout()

# OpenCV Harris
plt.subplot(2, 2, 2)
plt.imshow(sample_img, cmap = 'gray')
plt.title('OpenCV Harris Feature Detection')
plt.axis('off')
plt.scatter(o_x, o_y, s = 5, c = 'red')
plt.tight_layout()

# Overlap of feature points
plt.subplot(2, 2, 3)
plt.imshow(sample_img, cmap = 'gray')
plt.scatter(m_x, m_y, s = 5)
plt.scatter(o_x, o_y, s = 5, c = 'red')
plt.title('Overlap of feature points')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{harris_comparison_demo_img_path}/harris_comparison_demo.png', dpi = 300, bbox_inches = 'tight')



