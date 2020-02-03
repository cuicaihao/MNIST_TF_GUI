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
best_model_filepath =  "mnist.weights.best.2019.09.06_15.21.36.h5"
# best_model_filepath = ''
model =   load_model(best_model_filepath)
model.summary()
 
score = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Train loss: 0.03422381600427131
# Train accuracy: 0.9896833333333334
# Test loss: 0.03982059933549608
# Test accuracy: 0.9865

#%%

epochs = 10
batch_size = 1024

now = datetime.now()
best_model_filepath="mnist.weights.best."+str( now.strftime("%Y.%m.%d_%H.%M.%S"))+".h5"
checkpoint = ModelCheckpoint(best_model_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print('*'*80)
print('Start the training process at:', datetime.now())
print('*'*80)
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)

print('*'*80)
print('End the training process at:', datetime.now())
print('*'*80)
# save the training records
pickle_out = open("history_retrain.pickle","wb")
pickle.dump(history, pickle_out)

now=datetime.now()
save_model_dir = './models/'
final_model_name = str( now.strftime("%Y.%m.%d_%H.%M.%S")) +'.MnistModel.h5'
save_final_model_path = os.path.join(save_model_dir, final_model_name)
model.save(save_final_model_path)

#%%
plt.figure(figsize=(8, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('retraining_validation_accuracy_values.png', dpi= 300 )

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('retraining_validation_loss_values.png', dpi= 300 )


#%%
