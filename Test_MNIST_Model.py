# -*- coding: utf-8 -*-

"""
Created on Friday 2 Feb 2020

@author: Chris.Cui

Email: Chris.Cui@aurecongroup.com

"""
#%% load all the pkgs
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
from keras.utils import to_categorical

import numpy as np 

#%% Load the data
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
#%% Data preprocessing
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%%
best_model_filepath =  r"./models/mnist.weights.best.h5"
Model =   load_model(best_model_filepath)
Model.summary()

#%%
score_train = Model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])

score_test = Model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])


#%% check results
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
 
    
#%% make a single prediction
random_id = np.random.randint(len(x_train)) 
x = x_train[random_id,:,:,:] 
plt.figure()
plt.imshow(np.squeeze(x)) # remove extra dimensions
plt.colorbar()
plt.grid(False)
plt.show()

x = np.expand_dims(x, axis=0)
y = Model.predict(x)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('output vector {}'.format(y))
print('label: {}'.format(np.argmax(y)))

#%% make a batch prediction to get comfusion matrix
y_train_label = np.argmax(y_train, axis=1)
y_train_predic = Model.predict(x_train, verbose=1)
y_train_predic_label = np.argmax(y_train_predic, axis=1)
cm_train = confusion_matrix(y_train_label, y_train_predic_label)
print(cm_train)
plt.imshow(cm_train )

#%%  testing comfusion matrix
y_test_label = np.argmax(y_test, axis=1)
y_test_predic = Model.predict(x_test, verbose=1)
y_test_predic_label = np.argmax(y_test_predic, axis=1)
cm_test = confusion_matrix(y_test_label, y_test_predic_label)
print(cm_test)
plt.imshow(cm_test )

#%% precision recall 
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

print("Train: label precision recall")
cm = cm_train
for label in range(10):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))


print("Tests label precision recall")
cm = cm_test
for label in range(10):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))

 
#%% Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

print("Train accuracy total:", accuracy(cm_train))
print("Train accuracy total:", accuracy(cm_test))


 

# %%
