import numpy as np


def transform_xy(X, Y, inner, outer):

    (M, N) = (64, 512)
    theta = 2 * np.pi * X / N

    x1 = inner[0]
    y1 = inner[1]
    r1 = inner[2]

    x2 = outer[0]
    y2 = outer[1]
    r2 = outer[2]

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])

    distance = np.linalg.norm(p1-p2)

    diff_angle = np.arctan(((y2-y1) / (x2-x1)))
    long_r = (2 * distance * np.cos(diff_angle) +
              np.sqrt((2 * distance * np.cos(diff_angle)) ** 2 - 4 * (distance**2 - r2**2))) / 2

    x_i = x1 + r1 * np.cos(theta)
    y_i = y1 + r1 * np.sin(theta)

    x_o = x1 + long_r * np.cos(theta)
    y_o = y1 + long_r * np.sin(theta)

    x = int(x_i + (x_o - x_i) * Y / M)
    y = int(y_i + (y_o - y_i) * Y / M)

    x = min(319, x) or max(0, x)
    y = min(279, y) or max(0, y)

    return x, y


def normalize_image(image, inner, outer):

    new = np.zeros((64, 512))
    for Y in range(64):
        for X in range(512):
            x, y = transform_xy(X, Y, inner, outer)
            new[Y, 511-X] = image[y, x]

    return new


def rotate(image, degree):
    # Calculate the number of pixels to shift based on the degree of rotation

    pixels = abs(int(image.shape[1] * degree / 360))

    # Positive degrees roll to the right, negative to the left.

    rotated_image = np.roll(image, shift=pixels if degree > 0 else -pixels, axis=1)

    return rotated_image


