'''
This file demonstrates Shi-Tomasi Feature Detection which is an improved Harris feature detection technique
'''
# Importing necessary libraries
from config.paths import sample_img_path, shi_tomasi_demo_img_path
from src.image_processing.feature_detection.Shi_Tomasi_corner_detection import shi_tomasi
import cv2
import matplotlib.pyplot as plt
import os

# Loading in the image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# Getting the Shi-Tomasi Corner points tensor
shi_tomasi_corners = shi_tomasi(sample_img)
x , y = [], []
# corners.ravel(0) flattens the tensor into a list of tuples
for corner in shi_tomasi_corners:
    px, py = corner.ravel()
    x.append(px)
    y.append(py)

# Plotting the image
plt.imshow(sample_img, cmap = 'gray')
plt.scatter(x, y, c = 'red', s = 5)
plt.title('Shi-Tomasi corner points')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{shi_tomasi_demo_img_path}/Shi_Tomasi_demo.png', dpi = 300, bbox_inches = 'tight')
    