'''
This file demonstrates feature matching using BRIEF descriptor from ORB to check the rotational inavariance
'''
# Importing necessary libraries
import os
import matplotlib.pyplot as plt
from config.paths import sample_img_path, orb_feature_matching_demo_img_path
from src.image_processing.feature_detection.ORB_feature_matching import rotate_image, feature_matching
import cv2

# Loading the image
sample_img = cv2.imread(f'{sample_img_path}/{os.listdir(sample_img_path)[0]}', 0)

# Computing the rotated image
rot_sample_img = rotate_image(sample_img, 30)

# Getting the key points and descriptors and image matches
orig_sample_kp, orig_sample_desc, rot_sample_kp, rot_sample_desc, matches = feature_matching(sample_img, rot_sample_img)

# Sorting the matches according to the hamming distance
matches = sorted(matches, key = lambda x: x.distance)
# Creating an instance for drawing the matches
# Only getting the top 50 feature matches to avoid clutter
img_matches = cv2.drawMatches(sample_img, orig_sample_kp, rot_sample_img, rot_sample_kp, matches[: 50], None)

# Plotting the image matches
plt.figure(figsize = (10, 7))
plt.imshow(img_matches)
plt.title('ORB feature matches')
plt.axis('off')
plt.tight_layout()

plt.savefig(f'{orb_feature_matching_demo_img_path}/ORB_feature_match_demo.png', dpi = 300, bbox_inches = 'tight')
