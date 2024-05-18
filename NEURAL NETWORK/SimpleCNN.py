#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 08:03:12 2024

@author: cornelius
"""

# -----------------------------------------------------------------------------
#                                 IMPORTS
# -----------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
#                              DATA PREPARATION
# -----------------------------------------------------------------------------

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 
                                    train_images.shape[1],
                                    train_images.shape[1],
                                    1
                                    ).astype('float32')
print(train_images.shape) # added a color channel


# -----------------------------------------------------------------------------
#                        CONVOLUTIONAL NEURAL NETWORK
# -----------------------------------------------------------------------------


#                               1. Adding Layers

        #               -> Convolutional Layer
        #               -> Pooling Layer
        #               -> Flatten (for Dense Layer)
        #               -> Dense Layer (for classification)

        

model = tf.keras.Sequential()

model.add(Conv2D(filters=1,                         # amount of kernels to convolve with 
                 kernel_size=(5,5),                 # size of the kernel matrix
                 strides=(1,1),                     # step size for padding in x- and y-direction
                 padding='same',                    # padding such that input size matches output size
                 data_format = 'channels_last',     # data are in Format NHWC (#Samples, height, width, #Channels)
                 activation='relu'))                # activation function

model.add(MaxPool2D(pool_size=(2,2)))           
                                     
model.add(Flatten())

model.add(Dense(units=10, 
                activation='softmax'))


#                          2. Compilation and Training

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])                 # metrics evaluated during training

model.fit(train_images,train_labels, 
          epochs = 1, 
          validation_data = None, 
          shuffle=True)


# -----------------------------------------------------------------------------
#                                 VISUALIZATION
# -----------------------------------------------------------------------------

predictions = model.predict(test_images)  # numpy array

predicted_labels = [np.argmax(prediction) for prediction in predictions]

fig, axes = plt.subplots(2,6,figsize=(6,2))

for i in range(2):
    for j in range(6):
        
        axes[i][j].imshow(test_images[i*2+j],cmap='gray')
        axes[i][j].axis('off')
        axes[i][j].text(1,24,predicted_labels[i*2+j],color='yellow')
        
plt.show()































