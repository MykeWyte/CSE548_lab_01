# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang

Updated on Sat Jul 6 20:36:17 2024

@author: modified by Michael White 
"""

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

# Variable Setup
# Available datasets: KDDTrain+.txt, KDDTest+.txt, etc. More read Data Set Introduction.html within the NSL-KDD dataset folder
# Type the training dataset file name in ''
TrainingDataPath='NSL-KDD/'
TrainingData='KDDTrain+_20Percent.txt'
# Batch Size
BatchSize=10
# Epohe Size
NumEpoch=10


# Import dataset. path
# Dataset is given in TraningData variable You can replace it with the file path

# Michael White: Importing training datasets for scenarios A, B, and C
a_train_dataset = pd.read_csv("SATrain.csv", header=None)
a_test_dataset = pd.read_csv("SATest.csv", header=None)
b_train_dataset = pd.read_csv("SBTrain.csv", header=None)
b_test_dataset = pd.read_csv("SBTest.csv", header=None)
c_train_dataset = pd.read_csv("SCTrain.csv", header=None)
c_test_dataset = pd.read_csv("SCTest.csv", header=None)

# Michael White: labeling normal and malicious traffic for NN training and testing
sa_train_X = a_train_dataset.iloc[:, 0:-2].values
sa_train_label_column = a_train_dataset.iloc[:, -2].values
sa_train_y = []
for i in range(len(sa_train_label_column)):
    if sa_train_label_column[i] == 'normal':
        sa_train_y.append(0)
    else:
        sa_train_y.append(1)
sa_train_y = np.array(sa_train_y)

sa_test_X = a_test_dataset.iloc[:, 0:-2].values
sa_test_label_column = a_test_dataset.iloc[:, -2].values
sa_test_y = []
for i in range(len(sa_test_label_column)):
    if sa_test_label_column[i] == 'normal':
        sa_test_y.append(0)
    else:
        sa_test_y.append(1)
sa_test_y = np.array(sa_test_y)

sb_train_X = b_train_dataset.iloc[:, 0:-2].values
sb_train_label_column = b_train_dataset.iloc[:, -2].values
sb_train_y = []
for i in range(len(sb_train_label_column)):
    if sb_train_label_column[i] == 'normal':
        sb_train_y.append(0)
    else:
        sb_train_y.append(1)
sb_train_y = np.array(sb_train_y)

sb_test_X = b_test_dataset.iloc[:, 0:-2].values
sb_test_label_column = b_test_dataset.iloc[:, -2].values
sb_test_y = []
for i in range(len(sb_test_label_column)):
    if sb_test_label_column[i] == 'normal':
        sb_test_y.append(0)
    else:
        sb_test_y.append(1)
sb_test_y = np.array(sb_test_y)

sc_train_X = c_train_dataset.iloc[:, 0:-2].values
sc_train_label_column = c_train_dataset.iloc[:, -2].values
sc_train_y = []
for i in range(len(sc_train_label_column)):
    if sc_train_label_column[i] == 'normal':
        sc_train_y.append(0)
    else:
        sc_train_y.append(1)
sc_train_y = np.array(sc_train_y)

sc_test_X = c_test_dataset.iloc[:, 0:-2].values
sc_test_label_column = c_test_dataset.iloc[:, -2].values
sc_test_y = []
for i in range(len(sc_test_label_column)):
    if sc_test_label_column[i] == 'normal':
        sc_test_y.append(0)
    else:
        sc_test_y.append(1)
sc_test_y = np.array(sc_test_y)

# Encoding categorical data (convert letters/words in numbers)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Michael White: Ensure that the NN can run tests because training and testing sets have the same shape.
combined_X = np.vstack((sa_train_X, sa_test_X, sb_train_X, sb_test_X, sc_train_X, sc_test_X))

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
ct.fit(combined_X)

# Michael White: transform categorical values to numerical values
sa_train_X = np.array(ct.transform(sa_train_X), dtype=np.float)
sa_test_X = np.array(ct.transform(sa_test_X), dtype=np.float)
sb_train_X = np.array(ct.transform(sb_train_X), dtype=np.float)
sb_test_X = np.array(ct.transform(sb_test_X), dtype=np.float)
sc_train_X = np.array(ct.transform(sc_train_X), dtype=np.float)
sc_test_X = np.array(ct.transform(sc_test_X), dtype=np.float)

# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is 
# MinMaxScaler. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Michael White: minmax scale all datasets
sa_train_X = sc.fit_transform(sa_train_X)
sa_test_X = sc.fit_transform(sa_test_X)
sb_train_X = sc.fit_transform(sb_train_X)
sb_test_X = sc.fit_transform(sb_test_X)
sc_train_X = sc.fit_transform(sc_train_X)
sc_test_X = sc.fit_transform(sc_test_X)


########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN

# Michael White: Initializing the ANN for scenarios A, B, and C
sa_classifier = Sequential()
sb_classifier = Sequential()
sc_classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables


# Michael White: Building structure of all 3 neural networks
sa_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(sa_train_X[0])))
sa_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
sa_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
sa_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

sb_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(sb_train_X[0])))
sb_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
sb_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
sb_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

sc_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(sc_train_X[0])))
sc_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
sc_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
sc_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.

# evaluate the keras model for the provided model and dataset

# Michael White: run and evaluate keras model for scenarios A, B, and C
sa_classifierHistory = sa_classifier.fit(sa_train_X, sa_train_y, batch_size = BatchSize, epochs = NumEpoch)
sa_loss, sa_accuracy = sa_classifier.evaluate(sa_train_X, sa_train_y)
print('Print the loss and the accuracy of the model on scenario A dataset')
print('Loss [0,1]: %.4f' % (sa_loss), 'Accuracy [0,1]: %.4f' % (sa_accuracy))

sb_classifierHistory = sb_classifier.fit(sb_train_X, sb_train_y, batch_size = BatchSize, epochs = NumEpoch)
sb_loss, sb_accuracy = sb_classifier.evaluate(sb_train_X, sb_train_y)
print('Print the loss and the accuracy of the model on scenario B dataset')
print('Loss [0,1]: %.4f' % (sb_loss), 'Accuracy [0,1]: %.4f' % (sb_accuracy))

sc_classifierHistory = sc_classifier.fit(sc_train_X, sc_train_y, batch_size = BatchSize, epochs = NumEpoch)
sc_loss, sc_accuracy = sc_classifier.evaluate(sc_train_X, sc_train_y)
print('Print the loss and the accuracy of the model on scenario C dataset')
print('Loss [0,1]: %.4f' % (sc_loss), 'Accuracy [0,1]: %.4f' % (sc_accuracy))

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix

print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')

# Michael White: making predictions and evaluating model for scenatios A, B, and C
sa_y_pred = sa_classifier.predict(sa_test_X)
sa_y_pred = (sa_y_pred > 0.90)   
sa_cm = confusion_matrix(sa_test_y, sa_y_pred)
print('Confusion Matrix of scenario A:')
print(sa_cm)

sb_y_pred = sb_classifier.predict(sb_test_X)
sb_y_pred = (sb_y_pred > 0.9)   
sb_cm = confusion_matrix(sb_test_y, sb_y_pred)
print('Confusion Matrix of scenario B:')
print(sb_cm)

sc_y_pred = sc_classifier.predict(sc_test_X)
sc_y_pred = (sc_y_pred > 0.9)   
sc_cm = confusion_matrix(sc_test_y, sc_y_pred)
print('Confusion Matrix of scenario C:')
print(sc_cm)
