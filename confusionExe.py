import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Define file paths
fits_dir = '/Users/ronaktoprani/Desktop/internship/fitsWide'
label_file = '/Users/ronaktoprani/Desktop/internship/labels/image.txt'

# Loading image data
galaxy_images = []
galaxy_labels = []

# Load multiple FITS files separately and append them to the list
try:
    fits_files = ['/Users/ronaktoprani/Desktop/internship/fitsWide/115.fits',
                  '/Users/ronaktoprani/Desktop/internship/fitsWide/150.fits',
                  '/Users/ronaktoprani/Desktop/internship/fitsWide/200.fits',
                  '/Users/ronaktoprani/Desktop/internship/fitsWide/277.fits',
                  '/Users/ronaktoprani/Desktop/internship/fitsWide/444.fits']
    for file in fits_files:
        with fits.open(file) as hdul:
            galaxy_images.append(hdul[1].data)
            # Load corresponding labels for each image
            label = file.split('/')[-1].split('.')[0]  # Extract label from file name
            galaxy_labels.append(label)
except Exception as e:
    print("Error loading FITS files:", e)
    exit()

print("Number of loaded FITS images:", len(galaxy_images))

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
galaxy_labels_encoded = label_encoder.fit_transform(galaxy_labels)

# Verify that the number of labels matches the number of images
if len(galaxy_labels_encoded) != len(galaxy_images):
    print("Error: number of labels does not match the number of images")
    exit()

# Extract features from images
galaxy_features = []

# Extract HOG features from images
max_len = 0
for image in galaxy_images:
    hog_feature = feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), transform_sqrt=True, feature_vector=True)
    if hog_feature.size > max_len:
        max_len = hog_feature.size
    galaxy_features.append(hog_feature)

print("Number of extracted features:", len(galaxy_features))

# Pad feature vectors with zeros so that they all have the same length
for i in range(len(galaxy_features)):
    hog_feature = galaxy_features[i]
    new_feature = np.zeros(max_len)
    new_feature[:hog_feature.size] = hog_feature
    galaxy_features[i] = new_feature

# Convert galaxy_features to a numpy array
galaxy_features = np.array(galaxy_features)

print("Shape of features array:", galaxy_features.shape)

# Split data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(galaxy_features, galaxy_labels_encoded, test_size=0.2, random_state=42)
except ValueError as e:
    print("Error splitting data:", e)
    exit()

print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train an SVM classifier
clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train_imputed, y_train)

# Predict labels for test set
y_pred = clf.predict(X_test_imputed)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Decode labels
decoded_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(decoded_labels))
plt.xticks(tick_marks, decoded_labels, rotation=90)
plt.yticks(tick_marks, decoded_labels)
plt.show()