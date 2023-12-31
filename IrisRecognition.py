import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal

from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *
from IrisMatching      import *
from IrisPerformanceEvaluation import *



def main():
    runAll()


# This function would run all the algorithm step by step including IrisLocalization,
# IrisNormalization, IrisEnhancement, IrisFeatureExtraction, IrisMatching, 
# and IrisPerformanceEnvaluation.

# In addition to the LDA plot required by the project, I did PCA for dimension 
# reduction and ploted accuracy curve for different PCA dimensions

def runAll():
    # Run the algorithm for all training and testing images and save the result
    trainBase = create_rotated_training_data()
    testBase = create_test_data()
    irisTrain = np.array(trainBase)
    np.save('irisTrain',irisTrain)
    irisTest = np.array(testBase)
    np.save('irisTest',irisTest)
    
    # After transfering the image into vector, get performance envaluation by
    # calculating Acuracy curve for different PCA dimension reduction,
    # CRR Curve, and recognition results tables.
    train = np.load('irisTrain.npy')
    test = np.load('irisTest.npy')
    
    # Plot accuracy curve for different dimension reduction using PCA
    getPCACurve(train,test)
    
    # Plot accuracy curve for different dimensionality of the LDA
    getCRRCurve(train,test)
    
    # Draw a table for recognition results using different similarity measures
    getTable(train,test)

def runAll_test():

    try:
        # After transferring the image into a vector, get performance evaluation by
        # calculating Accuracy curve for different PCA dimension reduction,
        # CRR Curve, and recognition results tables.
        train = np.load('irisTrain.npy')
        test = np.load('irisTest.npy')
    except Exception as e:
        print(f"Error loading numpy arrays: {e}")
        raise
    
    try:
        # Plot accuracy curve for different dimensionality of the LDA
        getPCACurve(train, test)
    except Exception as e:
        print(f"Error during PCA curve generation: {e}")
        raise

# Call the main function
if __name__ == "__main__":
    main()