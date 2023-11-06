import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *
from IrisMatching import *


def table_CRR(self, train_features, train_classes, test_features, test_classes):
    # Assuming getMatching takes the features and classes as input
    # This method should generate a table of CRR based on different similarity measures
    vec = []
    dimensions = [100, 107]
    for i in range(1, 4):
        print(f'Currently computing distance measure number {i}')
        for dim in dimensions:
            vec.append(getMatching(train_features, test_features, LDA_dimension=dim, distance_measure=i))
    vec = np.array(vec).reshape(3, 2)
    df = pd.DataFrame(vec, index=['L1 distance measure', 'L2 distance measure', 'Cosine similarity measure'],
                        columns=['Original Feature Set', 'Reduced Feature Set'])
    print(df)
    return df

def performance_evaluation(self, train_features, train_classes, test_features, test_classes):
    # This method should plot the CRR curve for different PCA dimensions and LDA configurations
    vec = []
    dimensions = [50, 60, 70, 80, 90, 100, 107]
    plt.figure()
    for dim in dimensions:
        print(f'Currently computing dimension {dim}')
        crr_value = getMatching(train_features, test_features, LDA_dimension=dim)
        vec.append(crr_value)
    lw = 2
    plt.plot(dimensions, vec, color='darkorange', lw=lw)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate')
    plt.title('Recognition results using features of different dimensionality')
    plt.scatter(dimensions, vec, marker='*')
    plt.show()

def FM_FNM_table(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u,thresholds):
    print ("False Match and False Nonmatch Rates with Different Threshold Values")
    print tabulate([[thresholds[7], str(fmrs_mean[7])+"["+str(fmrs_l[7])+","+str(fmrs_u[7])+"]",str(fnmrs_mean[7])+"["+str(fnmrs_l[7])+","+str(fnmrs_u[7])+"]"], 
                    [thresholds[8], str(fmrs_mean[8])+"["+str(fmrs_l[8])+","+str(fmrs_u[8])+"]",str(fnmrs_mean[8])+"["+str(fnmrs_l[8])+","+str(fnmrs_u[8])+"]"],
                    [thresholds[9], str(fmrs_mean[9])+"["+str(fmrs_l[9])+","+str(fmrs_u[9])+"]",str(fnmrs_mean[9])+"["+str(fnmrs_l[9])+","+str(fnmrs_u[9])+"]"]],
                   headers=['Threshold', 'False match rate(%)',"False non-match rate(%)"])
#FM_FNM_table(train_features, train_classes, test_features, test_classes, thresholds_2)

def FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_l, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_u, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 60])
    plt.ylim([0.0,40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FMR Confidence Interval')
    plt.savefig('figure_13_a.png')
    plt.show()
    
def FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_mean, fnmrs_l, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_mean, fnmrs_u, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100])
    plt.ylim([0.0,40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FNMR Confidence Interval')
    plt.savefig('figure_13_b.png')
    plt.show()

    