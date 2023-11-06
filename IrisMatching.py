# import needed libraries
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy
# import needed functions from other files
from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *

# create a function that converts an image file name to a vector to be used by the matching model
def process_iris(file_name):
    # read the file
    raw_image = cv2.imread(file_name)
    # convert to grayscale
    gray_img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY) 
    # create localization array with iris_localize function
    (inner_circle, outer_circle) = Iris_localize(gray_img)
    # normalize the image using normalize_image function
    normalized_img = normalize_image(gray_img, inner_circle, outer_circle)
    # enhance the image with iris_enhancement function
    enhanced_img = iris_enhancement(normalized_img)
    # filter image twice to get feature vector
    (filtered_im1 , filtered_im2) = (filtered_image(enhanced_img, 3,1.5),filtered_image(enhanced_img, 4.5,1.5) )
    feature_vector = get_feature_vector(filtered_im1, filtered_im2)
    return feature_vector

# create the data frame for training data and testing data




def IrisMatching(training_data ,testing_data ,LDADimention=107,distanceMeasure=3):
    X_train = np.array(training_data)
    X_test  = np.array(testing_data)
    irisY = np.arange(1,109)
    Y_train = np.repeat(irisY,3*7)
    Y_test = np.repeat(irisY,4)
    trainClass = np.repeat(irisY,3)
    
    clf = LDA(n_components = LDADimention)
    clf.fit(X_train,Y_train)
    newTrain = clf.transform(X_train)
    newTest = clf.transform(X_test)
    
    
    predicted = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        vec = np.zeros(int(X_train.shape[0]/7))
        thisTest = newTest[i:i+1]
        for j in range(len(vec)):
            distance = np.zeros(7)
            for q in range(7):
                if distanceMeasure ==3:
                    distance[q] = scipy.spatial.distance.cosine(thisTest,newTrain[j*7+q:j*7+q+1])
                elif distanceMeasure ==1:
                    distance[q] = scipy.spatial.distance.cityblock(thisTest,newTrain[j*7+q:j*7+q+1])
                else:
                    distance[q] = scipy.spatial.distance.sqeuclidean(thisTest,newTrain[j*7+q:j*7+q+1])
                
            vec[j] = np.min(distance)
        shortestDistanceIndex = np.argmin(vec)
        predicted[i] = trainClass[shortestDistanceIndex]
    
    predicted = np.array(predicted,dtype =np.int)
    accuracyRate = 1 - sum(predicted != Y_test)/len(Y_test)
    return accuracyRate
