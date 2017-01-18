import csv
import pdb
import math
import pprint
import numpy
import matplotlib.pyplot as plt
import sys

# Data Processing
########## Retrieve and store training data ###########
label_list = []
with open(sys.argv[2], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        label_list.append(int(row[0]))

train_Y = list(label_list)

for row_index in range(0, len(label_list)):
    tup = (label_list[row_index], row_index)
    label_list[row_index] = tup

########## Retrieve and store training data ###########
test_Y = []
with open('testY.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_Y.append(int(row[0]))

########## Retrieve, store and restructure training data ###########
train_X = []
with open(sys.argv[1], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    index_counter = 0
    for row in reader:
        floatArray = []
        for i in row:
            floatArray.append(float(i))
        tup = (floatArray, index_counter)
        train_X.append(tup) # 455 x 30
        index_counter += 1

############### Reading and processing the test data ###############
test_X = []
with open(sys.argv[3], 'rb') as csvfile:
    reader = csv.reader(csvfile)
    index_counter = 0
    for row in reader:
        floatArray = []
        for i in row:
            floatArray.append(float(i))
        tup = (floatArray, index_counter)
        test_X.append(tup)
        index_counter += 1

####################################################################

# Constants
DECIMAL_VALUES = 8
NUM_FEATURES = len(train_X[0][0]) # 30
NUM_EXAMPLES = len(train_X) # 455

# Classes
class Node:
    nodeIndex = 0
    def __init__(self, isLeaf, label):
        self.nodeID = Node.nodeIndex
        Node.nodeIndex += 1
        self.leftNode = None
        self.rightNode = None
        self.zero = []
        self.one = []
        self.isLeaf = isLeaf
        self.label = label

        self.feature_index = None # feature_index
        self.threshold_value = None # threshold value

class DecisionTree:
    def __init__(self):
        self.topNode = None

    def hasLeafProperties(self, dataset):
        # Base Case 1: stop recurs when there is no data in this branch / node
        if len(dataset) == 0:
            return True, -1

        # Base Case 2: stop recurs when all labels in the subset are the same
        index_of_first_label = dataset[0][1]
        first_label = label_list[index_of_first_label][0]
        for row_index in range(0, len(dataset)):
            index_of_data = dataset[row_index][1]
            label = label_list[index_of_data][0]
            if label != first_label:
                return False, -1

        return True, first_label

    def makePredictions(self, test_X):
        predictionArray = []
        for feature_values, ind in test_X:
            pushLabel(predictionArray, self.topNode, feature_values, ind)

        return predictionArray

    def testTree(self,TEST_X,TEST_Y):
        predictions = self.makePredictions(TEST_X)

        correct = 0
        for value, index in predictions:
            if value == TEST_Y[index]:
                correct += 1

        accuracy = round(float(correct) * 100 / float(len(predictions)), DECIMAL_VALUES)
        return accuracy

    def learnTree(self, dataset):
        self.topNode = self.splitNode(dataset)

    def splitNode(self, dataset):
        # check with edge cases to see if stop recurs
        isLeaf, label = self.hasLeafProperties(dataset)
        if isLeaf:
            return Node(True, label)

        # find best configuration
        best_feature_index, best_threshold_index, threshold_values = self.findBestFeature(dataset)

        # split dataset
        left_data = []
        right_data = []
        for example in dataset:
            index = example[1]
            if example[0][best_feature_index] <= threshold_values[best_threshold_index]:
                left_data.append(example)
            else:
                right_data.append(example)

        # since this node splits, it is not a leaf
        node = Node(False, -1)
        node.leftNode = self.splitNode(left_data)
        node.rightNode = self.splitNode(right_data)
        node.feature_index = best_feature_index
        node.threshold_value = threshold_values[best_threshold_index]

        return node

    def findBestFeature(self, dataset):
        # Best configurations
        best_feature_index = -1
        best_threshold_index = -1
        best_information_gain = 0
        threshold_vals = []

        # Get index values for all features
        for feature_index in range(0, NUM_FEATURES): # 0 to 30
            # Get all 455 values for particular feature
            feature_values = []
            threshold_values = []
            for row_ind, record_tuple in enumerate(dataset): # 0 to 455
                sample_feature = record_tuple[0][feature_index]
                pair = (sample_feature, row_ind) # NOTE: access the value using [0]
                feature_values.append(pair)

            copy_feature_values = list(feature_values)
            copy_feature_values.sort(key=lambda tup: tup[0])

            # Get all 454 threshold values
            # Threshold values are chosen as average of two feature values.
            # Given [x1,x3,x5] threshold values would be x2=(x1+x3)/2, x4=(x3+x5)/2
            for row_index in range(0, len(dataset) - 1): # 0 to 454, -1 to not over-index
                lower_feature_value = copy_feature_values[row_index][0]
                higher_feature_value = copy_feature_values[row_index + 1][0]
                # threshold_value = (lower_feature_value + higher_feature_value) / float(2)
                threshold_value = (0.001 * lower_feature_value) + (0.999 * higher_feature_value)
                threshold_values.append(round(threshold_value, DECIMAL_VALUES))

            # For temporary storage
            leftNode = Node(False, -1)
            rightNode = Node(False, -1)

            # Loop through all 454 threshold values and push to left or right child nodes accordingly
            for threshold_index, threshold_value in enumerate(threshold_values): # 0 to 454
                for ind, feature_value in enumerate(feature_values):
                    feature_value = feature_value[0]
                    if feature_value <= threshold_value:
                        if label_list[ind][0] == 0:
                            leftNode.zero.append(feature_value)
                        else:
                            leftNode.one.append(feature_value)
                    else:
                        if label_list[ind][0] == 0:
                            rightNode.zero.append(feature_value)
                        else:
                            rightNode.one.append(feature_value)

                # Calculate information gain (IG) for this configuration
                information_gain = getInformationGain(leftNode, rightNode)

                # Remember and overwrite configuration for this information gain
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold_index = threshold_index
                    threshold_vals = threshold_values

        return best_feature_index, best_threshold_index, threshold_vals

# Helper Functions
def log(x):
    if x == 0.0:
        return 0
    else:
        return math.log(x, 2)

def probability_of_Y(value, left_ones, left_zeros, right_ones, right_zeros):
    probability = 0
    total = float(left_ones + left_zeros + right_ones + right_zeros)
    if value == '0':
        probability = float(left_zeros + right_zeros) / total
    else:
        probability = float(left_ones + right_ones) / total

    return probability

def entropy_of_Y(probability_one, probability_zero):
    entropy = (-1.0) * (probability_zero * log(probability_zero) + probability_one * log(probability_one))
    return entropy

def conditional_entropy_of_Y(left_ones, left_zeros, right_ones, right_zeros):
    left_total = float(left_ones + left_zeros)
    right_total = float(right_ones + right_zeros)
    total = float(left_ones + left_zeros + right_ones + right_zeros)

    probability_left = left_total / total if total != 0 else 0
    probability_right = right_total / total if total != 0 else 0
    probability_left_ones = left_ones / left_total if left_total != 0 else 0
    probability_left_zeros = left_zeros / left_total if left_total != 0 else 0
    probability_right_ones = right_ones / right_total if right_total != 0 else 0
    probability_right_zeros = right_zeros / right_total if right_total != 0 else 0

    conditional_entropy = (-1.0) * (probability_left * (probability_left_ones * log(probability_left_ones) \
                                                          + probability_left_zeros * log(probability_left_zeros)) \
                                     + probability_right * (probability_right_ones * log(probability_right_ones) \
                                                               + probability_right_zeros * log(probability_right_zeros)))

    return conditional_entropy

def getInformationGain(leftNode, rightNode):
    probability_of_Y_zero = probability_of_Y('0', len(leftNode.one), len(leftNode.zero), len(rightNode.one), len(rightNode.zero))
    probability_of_Y_one = 1.0 - probability_of_Y_zero
    entropy_Y = entropy_of_Y(probability_of_Y_one, probability_of_Y_zero)
    cond_entropy_Y = conditional_entropy_of_Y(len(leftNode.one), len(leftNode.zero), len(rightNode.one), len(rightNode.zero))

    return entropy_Y - cond_entropy_Y

def pushLabel(predictions, node, feature_values, ind):
    # Recursively find label prediction for particular feature_value
    if node.isLeaf:
        # if leaf node just append the tuple to the predictions array
        tup = (node.label, ind)
        predictions.append(tup)
    else:
        best_feature_index = node.feature_index
        feature_value = feature_values[best_feature_index]
        threshold = node.threshold_value

        if feature_value <= threshold:
            pushLabel(predictions, node.leftNode, feature_values, ind)
        else:
            pushLabel(predictions, node.rightNode, feature_values, ind)

    return

############ MAIN CODE #############

tree = DecisionTree()
tree.learnTree(train_X)

# Question 1 & 2
accuracy = tree.testTree(test_X, test_Y)
print "accuracy:", accuracy

# Write predictions to new file PredictY.csv
with open('PredictY.csv', 'wb') as csvfile:
    a = csv.writer(csvfile, delimiter=',')

    predictions = tree.makePredictions(test_X)
    # Reformate predictions array
    reformatted = []
    for value, ind in predictions:
        temp = []
        temp.append(value)
        reformatted.append(temp)

    a.writerows(reformatted)

# Question 3
test_sizes = numpy.arange(0.1, 1.1, 0.1)
test_accuracy_list = []
training_accuracy_list = []
for test_size in test_sizes:
    # Format test_size nicely, eg. 45.5 -> 45 examples
    num_examples = int(math.floor(float(test_size * len(train_X))))
    dataset = train_X[:num_examples]

    # Learn tree using dataset of specific size
    tree.learnTree(dataset)

    # Get accuracy of tree on test data
    accuracy = tree.testTree(test_X, test_Y)
    tup = (accuracy, round(test_size,1), num_examples)
    test_accuracy_list.append(tup)

    # Get accuracy of tree on training data
    training_accuracy = tree.testTree(train_X, train_Y)
    tup1 = (training_accuracy, round(test_size,1), num_examples)
    training_accuracy_list.append(tup1)

# Plot the results for training and test accuracies
x_axis = [size * 100 for size in test_sizes]
test_y_axis = [d[0] for d in test_accuracy_list]
training_y_axis = [d[0] for d in training_accuracy_list]

plt.plot(x_axis, test_y_axis, 'r--', x_axis, training_y_axis, 'b')
plt.xlabel('Size of dataset as a percentage of original size (%)')
plt.ylabel('Accuracy (%)')
plt.title('Comparison between training and testing accuracies with varying number of training samples')
plt.grid(True)
plt.show()
