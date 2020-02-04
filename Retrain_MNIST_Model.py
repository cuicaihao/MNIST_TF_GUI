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
from keras.callbacks import ModelCheckpoint
 
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pickle


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
# load the moded you want to boost the performance. 
best_model_filepath =  "./models/mnist.weights.best.h5"

# load the model from last time.
model =   load_model(best_model_filepath)
model.summary()
 
score = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Train loss: 0.015496265928385159
# Train accuracy: 0.9957333333333334
# Test loss: 0.027051866476310533
# Test accuracy: 0.9921

#%% retrain the model 
# change the optimizer in some way.
# keras.optimizers.Adadelta(),
# keras.optimizers.Adagrad(learning_rate=0.01)
print("New optimizer!!!")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adagrad(),
              metrics=['accuracy'])
print(model.summary())

#%%
batch_size = 1024
epochs = 30
retrain_model_filepath="./models/retrain.mnist.weights.best.h5"
checkpoint = ModelCheckpoint(retrain_model_filepath, 
                            monitor='val_acc', 
                            verbose=1, save_best_only=True, mode='Max')
stoppoint = EarlyStopping(monitor='val_acc', min_delta=0.001,
                            verbose=1, mode='Max', patience=5)

callbacks_list = [checkpoint, stoppoint]

print('*'*80)
print('Start the retraining process at:', datetime.now())
print('*'*80)
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)

print('*'*80)
print('End the retraining process at:', datetime.now())
print('*'*80)
# save the training records
pickle_out = open("./models/history_retrain.pickle","wb")
pickle.dump(history, pickle_out)

#%%
score_train = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])

score_test = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])

# compare with the orignal model's performance 
# Train loss: 0.008936051814166906
# Train accuracy: 0.9977333333333334 > 0.9957333333333334
# Test loss: 0.025634761017427808
# Test accuracy: 0.9922 > 0.9921 # accuracy imporved 0.0001

#%%
plt.figure(figsize=(8, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('./images/retraining_validation_accuracy_values.png', dpi= 300 )

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('./images/retraining_validation_loss_values.png', dpi= 300 )
 

# %%
