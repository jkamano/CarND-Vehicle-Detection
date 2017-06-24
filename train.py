#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:36:50 2017

@author: joao.ferreira
"""

import cv2
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

#Load Data and labels
def load_data(vehicle_folder="data/vehicles/*/*.png", nonvehicle_folder="data/non-vehicles/*/*.png"):
    v_paths = glob.glob(vehicle_folder)
    n_paths = glob.glob(nonvehicle_folder)
    features = v_paths + n_paths
    tag = np.append(np.ones(len(v_paths)), np.zeros(len(n_paths)))
    return (features, tag)
    
#Get features
def extract_features(features_list, cspace='CrCb', spatial_size=(16, 16),
                    hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, vis=False,
                    spatial_f=True, hog_f=True, hist_f=True):
    # Create a list to append feature vectors to
    features = []
    for i in tqdm(range(len(features_list))):
        f = features_list[i]
        
        # Read in each one by one
        image = mpimg.imread(f)
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'CrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        img_features = []
        if spatial_f:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        else:
            spatial_features = []
        if hist_f:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins)  
            img_features.append(hist_features)
        else:
            hist_features = []
        if hog_f:
            # Apply hog() to get spatial color features
            hog_features1 = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, ch=0)
            hog_features2 = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, ch=1)
            hog_features3 = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, ch=2)
            hog_features=np.hstack((hog_features1,hog_features2,hog_features3))
            img_features.append(hog_features)
        else:
            hog_features = []
        
        # Append the new feature vector to the features list
        features.append(np.concatenate(img_features))
    
    # Create an array stack of feature vectors
    X = np.vstack(features).astype(np.float64)
                       
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    features_scl = X_scaler.transform(X)
    
    if vis:
        print('# Spatial: ',len(spatial_features))
        print('# histogram: ', len(hist_features))
        print('# hog: ', len(hog_features))
        print('# Total: ', len(features[0]))
        fig = plt.figure(figsize=(5, 5))
        Nplots = 3
        # Original Image
        a=fig.add_subplot(3,Nplots,1)
        plt.imshow(image)
        #Transformed
        a=fig.add_subplot(3,Nplots,2)
        plt.imshow(feature_image[:,:,0],cmap='gray')
        a.set_title('Y')
        a=fig.add_subplot(3,Nplots,5)
        plt.imshow(feature_image[:,:,1],cmap='gray')
        a.set_title('Cr')
        a=fig.add_subplot(3,Nplots,8)
        plt.imshow(feature_image[:,:,2],cmap='gray')
        a.set_title('Cb')
        
        a=fig.add_subplot(3,Nplots,3)
        hh, pic = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=vis, feature_vec=True, ch=0)
        plt.imshow(np.copy(pic),cmap='gray')
        a.set_title('HOG 0')
        a=fig.add_subplot(3,Nplots,6)
        hh, pic = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=vis, feature_vec=True, ch=1)
        plt.imshow(np.copy(pic),cmap='gray')
        a.set_title('HOG 1')
        a=fig.add_subplot(3,Nplots,9)
        hh, pic = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, vis=vis, feature_vec=True, ch=2)
        plt.imshow(np.copy(pic),cmap='gray')
        a.set_title('HOG 2')
        plt.tight_layout()
        plt.show()
        

    # Return list of feature vectors
    return features_scl, X_scaler

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, ch=4):
    if vis == True:
        features, hog_image = hog(img[:,:,ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        if(ch==None):
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                           visualise=False, feature_vector=feature_vec)        
        elif(ch >= 3):
            features = []
            for ch in range(img.shape[2]):
                features.append(hog(img[:,:,ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                               visualise=False, feature_vector=feature_vec))
            features = np.ravel(features)
        else:
            features = hog(img[:,:,ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                           visualise=False, feature_vector=feature_vec)
        return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

if __name__ == "__main__":
    vis = True #Visualize hog features for a random sample
    train = False #If False subsamples the data set to speedup execution
    model_filename = "model.p"
    # Load available images
    img_list, tags = load_data()
    
    if vis or not train:
        # For test of algorithm take just 1% of all data
        x, img_list, y, tags = train_test_split(img_list, tags, test_size=0.01, random_state=np.random.randint(0, 100))
    
    # Extract 
    features, scaler = extract_features(img_list, vis=vis)
    X_train, X_test, y_train, y_test = train_test_split(features, tags, test_size=0.2, random_state=np.random.randint(0, 100))
    print('Complete Set of data instances: ',len(img_list))
    print('Size Training Set: ',len(X_train))
    print('Size Test Set: ',len(X_test))
    if train:
        # Use a linear SVC (support vector classifier)
        svc = LinearSVC()
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        pickle.dump((svc,scaler), open(model_filename, 'wb'))
    else:
        (svc,scaler) = pickle.load(open(model_filename, 'rb'))
        
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

