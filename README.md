# Binary Decision Tree

Implemented one of the common machine learning algorithms: Decision Trees. This python script will train and test a binary decision tree with the dataset as provided.

In designing this particular decision tree, I simply pick one feature to split on, and determine the threshold value to use in the split criterion for each non-leaf in the tree. The optimal split at each node is found using the information gain criterion. Since we are building a Binary Decision Tree (BDT), I will only do binary splits. Which means, each split should simply determine if the value of a particular feature in the feature vector of a sample is less than or equal to a threshold value or greater than the threshold value. Please note that the features in the provided dataset are continuously valued.

The main script is named BinaryDecisionTree.py which accepts 3 arguments and outputs the predictions in PredictY.csv. The code can be run in the terminal through using:

`python BinaryDecisionTree.py TrainX.csv TrainY.csv TestX.csv`

The code learns a BDT using the training set TrainX.csv and TrainY.csv, and then make predictions for all the samples in TestX.csv and output the labels to PredictY.csv.

Dataset:  

The dataset I use is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. We have split the date set into training set and test set stored in four csv files.

TrainX.csv has 455 samples, TrainY.csv has labels for the samples in TrainX.csv. Similarly, TestX.csv has 57 samples, TrainY.csv has labels for the samples in TrainX.csv. Each row in TrainX.csv or TestX.csv representing a sample of biopsied tissue. The tissue for each sample is imaged and 10 characteristics of the nuclei of cells present in each image are characterized. These characteristics are

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Number of concave portions of contour  
9. Symmetry
10. Fractal dimension

Each sample used in the dataset (TrainX.csv and TestX.csv) is a feature vector of length 30. The first 10 entries in this feature vector are the mean of the characteristics listed above for each image. The second 10 are the standard deviation and last 10 are the largest value of each of these characteristics present in each image.

Each sample is also associated with a label provided in TrainY.csv or TestY.csv. A label of value 1 indicates the sample was for malignant (cancerous) tissue. A label of value 0 indicates the sample was for benign tissue.
