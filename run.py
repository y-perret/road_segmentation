import numpy as np
import os,sys
from PIL import Image
from sklearn import neighbors
import re
import helpers as hp
import feature_extraction_functions as fef
from mask_to_submission import *


# Load the training set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(100, len(files)) # Load all the images of the training set
print("Loading " + str(n) + " images")
imgs = [hp.load_image(image_dir + files[i]) for i in range(n)]

# We increase the available training set with rotations
imgs = fef.data_augmentation(imgs)


gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground truth images")
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]

# We increase accordingly the groundtruth images with rotations
gt_imgs = fef.data_augmentation(gt_imgs)

n = len(imgs)

# patches of 4x4 pixels have been taken
patch_size = 4

# Get patches for images and groundtruth
img_patches = [hp.img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [hp.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# Extract all the features
X = np.asarray([fef.extract_features(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([hp.value_to_class_custom(gt_patches[i]) for i in range(len(gt_patches))])
print("Extracting all features done.")

# Create the classifier and train it
# Classifier implementing the k-nearest neighbors vote has been chosen
n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', p=1)
clf.fit(X, Y)

#Make submission file
test_dir = 'test_set_images/'
test_folders = os.listdir(test_dir)
test_folders = hp.natural_sort(test_folders)

save_dir = 'test_set_predictions/'


for i in range(len(test_folders)):
    test_sub_folders = os.listdir(test_dir + test_folders[i])
    test_img_name = test_dir + test_folders[i] + '/' + test_sub_folders[0]
    
    #create the features for the current image
    test_img = hp.load_image(test_img_name)
    img_size2 = test_img.shape[0] * test_img.shape[1]
    test_img_patches = hp.img_crop(test_img, patch_size, patch_size)
    X_test = np.asarray([fef.extract_features(test_img_patches[j]) for j in range(len(test_img_patches))])
    
    #Make the predictions
    Z_t = clf.predict(X_test)
    Z_img = hp.label_to_img(test_img.shape[0], test_img.shape[1], patch_size, patch_size, Z_t)
    
    #Save the image on disk
    result = Image.fromarray(hp.img_float_to_uint8(Z_img))
    result.save(save_dir + 'img_prediction_' + str(i+1) + '.png')
    

#Create the csv file
write_preds_to_csv()