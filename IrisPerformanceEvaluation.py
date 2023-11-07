

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *
from IrisMatching import *


# This function plots the recognition results using features of different 
# dimensionality
def getCRRCurve(train, test):
    # Assuming distance metrics are numbered 1, 2, and 3 for example purposes.
    distance_metrics = [1, 2, 3]
    dimension = [50, 60, 70, 80, 90, 100, 107]

    # Prepare the plot
    plt.figure()
    markers = ['o', 's', '*', 'x']  # Different markers for each distance metric
    colors = ['darkorange', 'navy', 'green', 'red']  # Different colors for each distance metric
    lw = 2

    # For each distance metric, calculate the CRRCurve
    for metric_idx, metric in enumerate(distance_metrics):
        vec = []
        for dim in dimension:
            print(f'Computing dimension {dim} with distance metric {metric}')
            # I'm assuming this function returns a value representing the correct recognition rate
            crr = IrisMatching_Rotation(rotated_training_data=train, testing_data=test, LDA_components=dim, distanceMeasure=metric)
            vec.append(crr)

        # Plot the results for this metric
        plt.plot(dimension, vec, color=colors[metric_idx], lw=lw, marker=markers[metric_idx], label=f'Distance Metric {metric}')

    # Configure and show the plot
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate')
    plt.title('Recognition results using features of different dimensionality')
    plt.legend()
    plt.show()

# Similar to getCRRCurve(), this function plots the accuracy rate for different
# dimensions for PCA. Within each PCA dimension, the maximum accuracy rate was 
# calculated by trying LDA dimensions of 90,100,107 which approves to be the dimensions
# with highest accuracy rate in general
def getPCACurve(train, test):
    train1 = train.copy()
    test1 = test.copy()
    
    pca_components = [400, 500, 600, 700, 800, 1000, 1200, 1400]
    lda_dimensions = [90, 100, 107]  # Assuming these are the LDA dimensions you want to plot
    
    plt.figure(figsize=(10, 6))
    
    # Loop over each LDA dimension
    for dim in lda_dimensions:
        vec = []
        for pca_comp in pca_components:
            thisPCA = PCA(n_components=pca_comp)
            thisPCA.fit(train1)
            train_pca = thisPCA.transform(train1)
            test_pca = thisPCA.transform(test1)
            
            # Compute the Iris Matching Rotation for the current dimension and PCA component
            recognition_rate = IrisMatching_Rotation(rotated_training_data=train_pca, testing_data=test_pca, LDA_components=dim, distanceMeasure=3)
            
            vec.append(recognition_rate)
        
        lw = 2
        # Plot the curve for this dimension
        plt.plot(pca_components, vec, lw=lw, label='LDA Dimension {}'.format(dim))
    
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Correct Recognition Rate')
    plt.title('Recognition Results using PCA with Different LDA Dimensions')
    
    # Add a legend to the plot
    plt.legend()

# This function prints the table of recognition results using different 
# similarity measures
def getTable(train,test):
    vec = []
    dimension = [100,107]
    for i in range(1,4):
        print('Currently computing distance measure number %d' %i)
        for dim in range(2):
            vec.append(IrisMatching_Rotation(train,test,LDA_components=dimension[dim],distanceMeasure=i))
    vec = np.array(vec).reshape(3,2)
    vec = pd.DataFrame(vec)
    vec.index = ['L1 distance measure', 'L2 distance measure','Cosine similarity measure']
    vec.columns = ['Original Feature Set', 'Reduced Feature Set']
    print(vec)
    return vec


def getPCA(train, test, pca_component=500):
    train1 = train.copy()
    test1 = test.copy()
    
    dimensions = [90, 100, 107]
    distance_metrics = [1, 2, 3]
    results = []

    # Apply PCA with 110 components
    pca = PCA(n_components=pca_component)
    pca.fit(train1)
    train_transformed = pca.transform(train1)
    test_transformed = pca.transform(test1)
    
    for dim in dimensions:
        for distance in distance_metrics:
            # Call the IrisMatching_Rotation function with the transformed data
            accuracy = IrisMatching_Rotation(rotated_training_data=train_transformed, testing_data=test_transformed, LDA_components=dim, distanceMeasure=distance)
            
            # Append the result for each LDA dimension and distance metric
            results.append({
                'LDA_Dimension': dim,
                'Distance_Metric': distance,
                'PCA': pca_component,
                'Accuracy': accuracy
            })

            print(f'PCA {pca_component} - LDA {dim} - Distance {distance} - Accuracy: {accuracy}')

    # Convert results to a pandas DataFrame for a nice table format
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

# Assume IrisMatching_Rotation is defined elsewhere and works as expected.
# This function would then be called with the training and testing datasets.

    