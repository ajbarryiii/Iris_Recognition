from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *


import numpy as np
import cv2
import scipy.signal
import os
from tqdm import tqdm

# Directory paths
input_dir = "/Users/ajbarry/Dropbox/My Mac (AJ’s MacBook Pro)/Downloads/Iris Recognition (1)/CASIA Iris Image Database (version 1.0)"
output_dir = "/Users/ajbarry/Dropbox/My Mac (AJ’s MacBook Pro)/Downloads/Iris Recognition (1)/Iris_localization_test"

import cv2
import os


# Replace 'path_to_your_image' with the path to the image you want to process
image_path = "/Users/ajbarry/Dropbox/My Mac (AJ’s MacBook Pro)/Downloads/Iris Recognition (1)/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was properly loaded
if img is None:
    raise ValueError(f"Failed to read image from {image_path}")

# Iris localization
circles = Iris_localize(img)

# Iris normalization
normalized_iris = normalize_image(img, circles[0], circles[1])

# Iris enhancement
enhanced_iris = iris_enhancement(normalized_iris)

# Ensure processing was successful
if enhanced_iris is None:
    raise ValueError("Iris enhancement failed.")

f1 = filter_image(enhanced_iris, 3,1.5)
f2 = filter_image(enhanced_iris, 4.5,1.5)

feature_vec = get_feature_vector(f1, f2)

print(feature_vec)