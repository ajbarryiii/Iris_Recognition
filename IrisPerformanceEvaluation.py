

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
    plt.title('Recognition results using PCA of different dimensionality')
    plt.legend()
    plt.show()

# This function plots the accuracy rate for different
# dimensions for PCA. Within each PCA dimension, the accuracy rate was 
# calculated by using LDA dimension 100
def getPCACurve(train,test):
    train1 = train.copy()
    test1 = test.copy()
    vec = []
    pca = [400,500,600,700,800,1000, 1200, 1400]
    dimension = [90, 100, 107]
    plt.figure()
    for p in range(len(pca)):
        thisPCA = PCA(n_components=pca[p])
        thisPCA.fit(train1)
        train = thisPCA.transform(train1)
        test  = thisPCA.transform(test1)
        for i in range(len(dimension)):
            ans = []
            print('Currently computing dimension %d' %dimension[i])
            ans.append(IrisMatching_Rotation(rotated_training_data=train,testing_data=test,LDA_components= dimension[i], distanceMeasure=3))
        vec.append(max(ans))
    lw = 2

    plt.plot(pca, vec, color='darkorange',lw=lw)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recgnition rate')
    plt.title('Recognition results using PCA of different dimensionality')
    plt.scatter(pca,vec,marker='*')

    plt.show()



# This function prints the table of recognition results using different 
# similarity measures and a fixed number of PCA components  
def getTable(train, test, pca_component=1100):
    train1 = train.copy()
    test1 = test.copy()
    
    dimensions = [90, 100, 107]
    distance_metrics = [1, 2, 3]
    results = []

    # Apply PCA with 1100 components
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


    