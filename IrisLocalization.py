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

    # Given Xp, Yp from previous computation
    half_size = 60
    roi = img[Yp - half_size:Yp + half_size, Xp - half_size:Xp + half_size]

    # Adaptive thresholding
    binary_roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Calculate the centroid
    M = cv2.moments(binary_roi)
    if M["m00"] != 0:
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Updating pupil coordinates based on centroid
    Xp_new = Xp - half_size + cX
    Yp_new = Yp - half_size + cY
    #Use a threshhold of 64 to localize the pupil
    region120 = img[Yp_new-60:Yp_new+60,Xp_new-60:Xp_new+60]
    ret,th1 = cv2.threshold(region120,64,65,cv2.THRESH_BINARY)

    #Based on the binary image, re-calculate the center of the pupil and estimate
    # the radius of the pupil
    mask1 = np.where(th1>0,1,0)

    vertical = mask1.sum(axis = 0)
    horizontal = mask1.sum(axis = 1)

    minyp = np.argmin(horizontal) 
    minxp = np.argmin(vertical)
    radius1 = (120 - sum(mask1[minyp])) / 2
    radius2 = (120 - np.sum(mask1,axis=0)[minxp]) / 2
    radius = int((radius1 + radius2) /2)

    #compute subimages
    region240 = img[np.arange(Yp_new-120, min(279, Yp_new+110)),:][:,np.arange(Xp_new-135,min(319,Xp_new+135))]
    region120 = img[np.arange(Yp_new-60, min(279, Yp_new+60)),:][:,np.arange(Xp_new-60,min(319,Xp_new+60))]


    for loop in range(1,5):
        circles = cv2.HoughCircles(region120,cv2.HOUGH_GRADIENT,1,250, param1=50,param2=10,minRadius=(radius-loop),maxRadius=(radius+loop))
        if type(circles) != type(None):
            break
        else:
            pass
    circles = np.around(circles)

    #gaussian blur for computing Iris boundary
    region240_blur = cv2.GaussianBlur(region240, (5, 5), 0)
    circles1 = cv2.HoughCircles(region240_blur, cv2.HOUGH_GRADIENT,1,250, param1=30,param2=10,minRadius=98,maxRadius=118)
    circles1 = np.around(circles1)                           


    image1 = img.copy()

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(image1,( int(i[0]+ Xp_new - 60),int(i[1] + Yp_new - 60)),int(i[2]),(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ Xp_new - 60),int(i[1] + Yp_new - 60)),int(i[2]),(0,255,0),2)
        innerCircle = [i[0] + Xp_new - 60,i[1] + Yp_new -60 ,i[2]]


    for i in circles1[0,:]:
        # draw the outer circle
        outerCircle = [int(i[0]+ Xp_new - 135),int(i[1] + Yp_new - 120),i[2]   ]
    
    # After computing the innerCircle and outterCircle...
    inner_x, inner_y, inner_r = innerCircle
    outer_x, outer_y, outer_r = outerCircle

    # Compute the distance between the centers of the two circles
    distance = np.sqrt((inner_x - outer_x)**2 + (inner_y - outer_y)**2)

    # Check if the outer circle's center is outside the inner circle
    if distance > 15:
        outerCircle[0] = inner_x
        outerCircle[1] = inner_y


    return(innerCircle,outerCircle)


