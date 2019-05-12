PCML Project 2: Road Segmentation

This project has been made in the case of a group project for an EPFL master course called Pattern Classification and Machine Learning.
In a group of three students, the goal was to use machine learning techniques learned in class to solve a problem of road segmentation of aerial images.
A set of training images (training\images folder) was provided with its corresponding ground truth (training\groundtruth folder) to train our methods.
The task was to experiment and test a lot of different methods in the hope of obtaining the best possible results.
The methods were finally applied on the test set (test_set_images folder) to get the results that were evaluated.

While this project is relatively small in terms of lines of code, it took time to think about possible solutions.
We had to try a lot of different approaches before finding our final solution.
Most of the tests ahev been executed on a Jupyter notebook for simplicity.

This project has been realized in 2017, meaning that I can't precisely remember on which part of the project I contributed the most.
However, since there was a lot of thinking to be made, we usually worked together on the same aspect brainstorming on which possibility we could explore.


This archive contains the following folders and file:

	training\images: training set of aerial images
	training\groundtruth: training set of binary images of segmented roads
    test_set_images: the images to test the model.
    test_set_predictions: the obtained predictions as binary images.
    
    feature_extraction_functions.py: contains all our methods we used to extract relevant features. Not all methods are used in the end.
    helpers.py: contains helper methods.
    run.py: the main script to run in a command prompt to get our best result. It trains a classifier using the training set of images and then tries to segments the roads for the testing set. The results are shown in the test_set_predictions folder.
    validations.py: contains our cross validation function to evaluate locally our approach
	
	svm_submission.csv: File used to submit our results online for evaluation
	
	The two following scripts were provided to submit our results
	mask_to_submission.py: provided script to convert mask to a csv submission.
	submission_to_mask.py: provided script to convert a cvs to mask.
    
For the machine learning tasks of this project we mainly used two libraries 
    sk-learn: http://scikit-learn.org/
    sci: http://scikit-learn.org/stable/
    
Our run.py script does not have the model saved, but the whole process should run in less than 10 minutes.