import cv2
import numpy as np

def iris_localization(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Step 1: Project the image in the vertical and horizontal direction
    vertical_projection = np.sum(img, axis=1)
    horizontal_projection = np.sum(img, axis=0)

    # Find the approximate pupil center coordinates (Xp, Yp)
    Yp = np.argmin(vertical_projection)
    Xp = np.argmin(horizontal_projection)

    # Step 2: Binarize a 120x120 region centered at (Xp, Yp)
    half_size = 60
    roi = img[Yp - half_size:Yp + half_size, Xp - half_size:Xp + half_size]
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the centroid of the binary region
    M = cv2.moments(binary_roi)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = half_size, half_size  # set centroid to the center of the ROI

    # Correct the centroid to the original image scale
    cX += Xp - half_size
    cY += Yp - half_size

    # Step 3: Calculate exact parameters using edge detection and Hough transform
    edges = cv2.Canny(img, 100, 200)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 120, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Show the result
    cv2.imshow('Iris Localization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

iris_localization("/Users/ajbarry/Dropbox/Documents/MA y2/Applied CV/Iris Recognition/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp")
