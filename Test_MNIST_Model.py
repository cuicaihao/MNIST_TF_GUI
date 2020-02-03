# -*- coding: utf-8 -*-

"""
Created on Friday 06-09-2019

@author: Chris.Cui

Email: Chris.Cui@aurecongroup.com

"""
#%% load all the pkgs
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K

import numpy as np 

#%% Load the data
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
#%% 
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
# best_model_filepath =  "mnist.weights.best.2019.09.06_11.31.49.h5"

best_model_filepath =  r"./models/2020.02.03_14.52.53.MnistModel.h5"

Model =   load_model(best_model_filepath)
Model.summary()


#%%
score_train = Model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])

score_test = Model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])


#%%
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
 
y_train_predic = Model.predict(x_train, verbose=1)

#%%
from keras.utils import to_categorical

#%%

y_train_label = np.argmax(y_train, axis=1)
y_train_predic_label = np.argmax(y_train_predic, axis=1)

#%%
from matplotlib import pyplot as plt

cm = confusion_matrix(y_train_label, y_train_predic_label)

import seaborn as sn
print(cm)
plt.imshow(cm )


#%%
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


print("label precision recall")
for label in range(10):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    
#%%

print("precision total:", precision_macro_average(cm))

print("recall total:", recall_macro_average(cm))


#%%

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

print("accuracy total:", accuracy(cm))
    
#%%
x = x_train[1,:,:,:]
x = np.expand_dims(x, axis=0)
y = Model.predict(x)
print(y)

 