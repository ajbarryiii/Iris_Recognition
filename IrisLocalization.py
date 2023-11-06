import cv2
import numpy as np

def Iris_localize(img):
    
    #Get maximum value in window
    window = 9
    kernel = np.ones((window, window),np.uint8)
    blur_max = cv2.dilate(img, kernel)

    # Compute the histogram of the image
    histogram_x = np.sum(blur_max, axis=0)
    Xp = np.argmin(histogram_x)

    histogram_y = np.sum(blur_max, axis=1)
    Yp = np.argmin(histogram_y)
    # Get the image dimensions
    height, width = img.shape

    # Ensure the region does not exceed image bounds
    x_start = max(Xp - 60, 0)
    x_end = min(Xp + 60, width)
    y_start = max(Yp - 60, 0)
    y_end = min(Yp + 60, height)

    # Update Xp, Yp to the center if the region is too small
    if (x_end - x_start) < 120 or (y_end - y_start) < 120:
        Xp = width // 2
        Yp = height // 2
        x_start = max(Xp - 60, 0)
        x_end = min(Xp + 60, width)
        y_start = max(Yp - 60, 0)
        y_end = min(Yp + 60, height)

    region120 = img[y_start:y_end, x_start:x_end]
    
    #Use a threshhold of 64 to localize the pupil
    ret,th1 = cv2.threshold(region120,64,255,cv2.THRESH_BINARY)

    #Based on the binary image, re-calculate the center of the pupil and estimate
    # the radius of the pupil
    
    mask1 = np.where(th1 > 0, 1, 0)
    
    vertical = mask1.sum(axis = 0)
    horizontal = mask1.sum(axis = 1)

    minyp = np.argmin(horizontal) 
    minxp = np.argmin(vertical)
    radius1 = (120 - sum(mask1[minyp])) / 2
    radius2 = (120 - np.sum(mask1,axis=0)[minxp]) / 2
    radius = max(int((radius1 + radius2) /2), 20)

    #Hough Circles for pupil
    for loop in range(1,5):
        circles = cv2.HoughCircles(region120,cv2.HOUGH_GRADIENT,1,250, param1=50,param2=10,minRadius=(radius-loop),maxRadius=(radius+loop))
        if type(circles) != type(None):
            break
        else:
            pass
    circles = np.around(circles)

    for i in circles[0,:]:
        #compute the pupil ciricle
        innerCircle = [int(i[0] + Xp - 60),int(i[1] + Yp -60) ,int(i[2])]


    #ROI for iris
    # Ensure the start and end are integers
    start_y = int(max(0, innerCircle[1] - 120))
    end_y = int(min(height, innerCircle[1] + 120))
    start_x = int(max(0, innerCircle[0] - 120))
    end_x = int(min(width, innerCircle[0] + 120))

    # ROI for iris
    region240 = img[start_y:end_y, start_x:end_x]

    #gaussian blur for computing Iris boundary
    region240_blur = cv2.GaussianBlur(region240, (5, 5), 0)
    circles1 = cv2.HoughCircles(region240_blur, cv2.HOUGH_GRADIENT,1,250, param1=30,param2=10,minRadius=98,maxRadius=118)
    circles1 = np.around(circles1)                           


    for i in circles1[0,:]:
        # compute the outer circle
        outerCircle = [int(i[0]+ innerCircle[0] - 135),int(i[1] + innerCircle[1] - 120),int(i[2])   ]
    
    # After computing the innerCircle and outterCircle...
    inner_x, inner_y, inner_r = innerCircle
    outer_x, outer_y, outer_r = outerCircle

    # Compute the distance between the centers of the two circles
    distance = np.sqrt((inner_x - outer_x)**2 + (inner_y - outer_y)**2)

    # Check if the outer circle's center is outside the inner circle
    if distance > 15:
        outerCircle[0] = inner_x + 1
        outerCircle[1] = inner_y + 1


    return(innerCircle,outerCircle)


