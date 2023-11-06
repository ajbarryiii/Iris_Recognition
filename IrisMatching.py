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
    (filtered_im1 , filtered_im2) = (filter_image(enhanced_img, 3,1.5),filter_image(enhanced_img, 4.5,1.5) )
    feature_vector = get_feature_vector(filtered_im1, filtered_im2)
    return feature_vector

# function for adding zeros so the database can be created
def add_leading_zeros(number):
    # Convert the number to a string
    number_str = str(number)
    
    # Calculate the number of zeros to add
    num_zeros_to_add = 3 - len(number_str)
    
    # Add leading zeros and return as a string
    result_str = '0' * num_zeros_to_add + number_str
    return result_str
# create the functions that create the data frame for training data and testing data
def create_training_data():
    # create an empty list to store the training data
    training_vector = []
    # iterate using a double for loop each training image file
    for i in np.arange(1,109):
        for j in np.arange(1,4):
            file = "./CASIA Iris Image Database (version 1.0)/" + add_leading_zeros(i) +"/1/" + add_leading_zeros(i)+ "_1_" + str(j)+ ".bmp"
            # return the processed feature vector and store it as processed vec
            processed_vec = process_iris(file)
            # add the processed_vec to the list
            training_vector.append(processed_vec)
    # return the list as the function output        
    return training_vector
# create a function that does the same for test data
def create_test_data():
    testing_vector = []
    for i in np.arange(1,109):
        for j in np.arange(1,5):
            file = "./CASIA Iris Image Database (version 1.0)/" + add_leading_zeros(i) +"/2/" + add_leading_zeros(i)+ "_2_" + str(j)+ ".bmp"
            processed_vec = process_iris(file)
            testing_vector.append(processed_vec) 
    return testing_vector

# create an IrisMatching function that takes in a training and testing data set and a number of LDA components and
# a distance measure 3= cosine measure, 1 = manhattan, and 2 = eucidean
def IrisMatching(training_data ,testing_data ,LDA_components,distanceMeasure):
    # make sure the training and testing data are np.arrays and not lists
    X_train = np.array(training_data)
    X_test  = np.array(testing_data)
    # create a Y variable called irisY that is the ID for each iris (from 1 to 108 in this data)
    irisY = np.arange(1,109)
    # create a Y_train vector that is the iris IDs 1-108 repeated three times as this is our training data
    Y_train = np.repeat(irisY,3)
    # do the same for Y_test, this time repeating each ID 4 times as we have 4 test images
    Y_test = np.repeat(irisY,4)
    trainClass = np.repeat(irisY,3)
    # perform LDA on the training data
    lda_model = LDA(n_components = LDA_components)
    lda_model.fit(X_train,Y_train)
    # transform (reduce) the training and testing data using the LDA model trained above and store them as Train_reduced and Test_reduced
    Train_reduced = lda_model.transform(X_train)
    Test_reduced = lda_model.transform(X_test)
    
    # create a np.array that is X_test.shape[0] rows long and call it predicted
    # this will store the predicted values
    predicted = np.zeros(X_test.shape[0])
    # initiate the for loop that goes through each row of the test dataset
    for i in range(X_test.shape[0]):
        # create a vector that is of the length of the number of training samples
        # this will store the distance each training sample is from the current test data
        dist_vector = np.zeros(int(X_train.shape[0]))
        # set the current test sample to be the feature vector of the current sample
        current_test_sample = Test_reduced[i]
        # for every training sample measure the distance of the current test data sample from that training data sample
        for j in range(len(dist_vector)):
            # if cosine distance is specified use the cosine distance
            if distanceMeasure ==3:
                distance = scipy.spatial.distance.cosine(current_test_sample,Train_reduced[j])
              # if l1 distance is specified use the l1 distance  
            elif distanceMeasure ==1:
                distance = scipy.spatial.distance.cityblock(current_test_sample,Train_reduced[j])
                # if eucidean distance is specified use the l2 distance 
            else:
                distance = scipy.spatial.distance.sqeuclidean(current_test_sample,Train_reduced[j])
           # store the distance in the distance vector
            dist_vector[j] = distance
        # set the prediction for the ith element of the test data to be the ID of the nearest training data    
        shortestDistanceIndex = np.argmin(dist_vector)
        predicted[i] = trainClass[shortestDistanceIndex]
    # set predicted to be a np.array
    predicted = np.array(predicted,dtype =np.int)
    # calculate accuracy rate
    accuracyRate = 1 - sum(predicted != Y_test)/len(Y_test)
    # return accuracy rate
    return accuracyRate

