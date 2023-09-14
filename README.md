# GalacticComponentMapping

## Introduction

This project focuses on applying machine learning techniques to map galactic components in the M74 galaxy using astronomical FITS image data. The code presented here demonstrates the entire pipeline, including data loading, preprocessing, and extracting Histogram of Oriented Gradients (HOG) features, training a support vector machine (SVM) classifier, and evaluating the model's performance

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Astropy
- scikit-image (skimage)
- scikit-learn (sklearn)
  
## Usage
Ensure that the FITS files are located in the fitsWide directory and the label file is located at the path specified in label_file variable in the code.

Run the Python script:


python galaxy_classification.py
The script will load the FITS files, extract HOG features, split the data into training and testing sets, train the SVM classifier, and evaluate the performance by calculating accuracy and generating a confusion matrix plot.


