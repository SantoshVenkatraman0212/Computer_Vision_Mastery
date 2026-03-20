'''
This file demonstrates the full harris feature detection pipeline
'''
# Importing necessary libraries
import numpy as np
import cv2
from config.paths import harris_demo_img_path, sample_img_path
from src.image_processing.feature_detection.structure_tensor import compute_structure_tensor
from src.image_processing.feature_detection.harris_response import compute_harris_response
from src.image_processing.feature_detection.harris import compute_harris
import os
import matplotlib.pyplot as plt

# Loading in the sample image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)
# Getting the structure tensors (Ixx, Iyy and Ixy)
Ixx, Iyy, Ixy = compute_structure_tensor(sample_img)
# Getting the harris Response
h_response = compute_harris_response(Ixx, Iyy, Ixy)
# Getting harris feature detection image
harris_feature_img = compute_harris(h_response)

# Creating a list of elements for plotting
plot_ele = [('Original Image', sample_img), ('Ixx', Ixx), ('Iyy', Iyy), ('Ixy', Ixy),  
            ('Harris feature detection image', harris_feature_img)]

# Normalizing the plots to maintain a uniform scale for comparison
plt.figure(figsize = (10, 7))
for index, ele in enumerate(plot_ele):
    plt.subplot(3, 3, index + 1)
    img = ele[1]
    if img.dtype == np.uint8:
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(((img - img.min()) / (img.max() - img.min())), cmap = 'gray')
    plt.title(f'{ele[0]}')
    plt.axis('off')
    plt.tight_layout()

# Plotting the harris response
plt.subplot(3, 3, 6)
plt.imshow(h_response, cmap = 'jet')
plt.title('Harris Response')
plt.axis('off')
plt.tight_layout()

# Now overlaying the harris feature img on the original image
plt.subplot(3, 3, 7)
if sample_img.dtype == np.uint8:
    plt.imshow(sample_img, cmap = 'gray')
else:
    plt.imshow((sample_img - sample_img.min()) / (sample_img.max() - sample_img.min()), 
               cmap = 'gray')

if harris_feature_img.dtype != np.uint8:
    harris_feature_img = (harris_feature_img - harris_feature_img.min()) / (
        harris_feature_img.max() - harris_feature_img.min())

# Getting only the coordinates where the pixel intensity changes in all directions
# i.e. getting only the white dots.
x, y = np.where(harris_feature_img > 0)
plt.scatter(x, y, s = 5)
plt.title('Harris feature overlay')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{harris_demo_img_path}/harris_demo_img.png', dpi = 300, bbox_inches = 'tight')

