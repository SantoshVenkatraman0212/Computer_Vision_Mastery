'''
This file demonstrates refined ORB feature matching using KNN-based matching 
and Lowe's ratio for optimal matches (strong)
'''
# Importing necessary libraries
import os
import matplotlib.pyplot as plt
from config.paths import sample_img_path, refined_orb_matching_demo_img_path
from src.image_processing.feature_detection.ORB_feature_matching import rotate_image, feature_matching
import cv2

# Loading the image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# Computing the rotated image
rot_sample_img = rotate_image(sample_img, 30)

# Getting the key points and descriptors and image matches
orig_sample_kp, orig_sample_desc, rot_sample_kp, rot_sample_desc, matches = feature_matching(sample_img, rot_sample_img, match_type = 'knn')

# Using Lowe's ratio as the threshold for getting the strong matches
# This results in a cleaner and more well defined and distinct feature matches
strong_matches = []
for i, j in matches:
    if i.distance < 0.75 * j.distance:
        strong_matches.append(i)

img_matches = cv2.drawMatches(sample_img, orig_sample_kp, rot_sample_img, rot_sample_kp, strong_matches[: 50], None)

# Plotting the image matches
plt.figure(figsize = (10, 7))
plt.imshow(img_matches)
plt.title('Refined ORB feature matches')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{refined_orb_matching_demo_img_path}/Refined_ORB_feature_match_demo.png', dpi = 300, bbox_inches = 'tight')