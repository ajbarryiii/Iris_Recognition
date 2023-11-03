import numpy as np
import cv2
import scipy.signal


def get_kernal(x, y, theta=0, l=10, psi=10, ksize=9):

    gamma = x / y
    kernel = cv2.getGaborKernel((ksize, ksize), x, theta, l, gamma, psi, ktype=cv2.CV_64F)
    return kernel


def filter_image(image, x, y):

    image = image[0:48, :]
    kernel = get_kernal(x, y)
    new = scipy.signal.convolve2d(image, kernel, mode='same')

    return new


def calculate_features(image, block_size=8):
    # Calculate the number of blocks in each dimension
    rows, cols = image.shape[:2]
    rows //= block_size
    cols //= block_size

    # Pre-allocate the feature vector
    features = np.zeros((rows, cols, 2), dtype=np.float64)

    # Vectorized computation over blocks
    for row in range(rows):
        for col in range(cols):
            block = image[row * block_size: (row + 1) * block_size,
                          col * block_size: (col + 1) * block_size]
            features[row, col, 0] = np.mean(np.abs(block))
            features[row, col, 1] = np.std(block)

    return features.ravel()


def get_feature_vector(f1, f2):
    return np.concatenate([calculate_features(f1), calculate_features(f2)])


