#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:49:05 2024

@author: cornelius
"""

# -----------------------------------------------------------------------------
#                               IMPORTS
# -----------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Reshape

# -----------------------------------------------------------------------------
#                           DATA PREPARATION
# -----------------------------------------------------------------------------

"""
        TRAINING DATA HAS SHAPE:   (#Samples, width_px, height_px, #Channels)
                                    
                                -> #Channels = 3 for RGB
                                -> #Channels = 1 for Grayscale

"""

# (train_images,_),(test_images,_) = tf.keras.datasets.cifar10.load_data()
(train_images,_),(test_images,_) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype('float32')/255.0 
test_images = test_images.astype('float32')/255.0


# -----------------------------------------------------------------------------
#                              AUTO ENCODER
# -----------------------------------------------------------------------------

"""
                -> Autoencoder is a deep neural network with a bottleneck
                   in the middle
                -> It consists of an Encoder and a Decoder
                -> The encoder takes the input data (the picture for example)
                   and reduces the dimensionality of it by transferring the
                   signal to the bottleneck
                -> The decoder takes the signal out of the bottleneck and
                   reconstructs the original input data out of this
                -> The amount of input neurons for a encoder is the same as the
                   amount of output neurons for a decoder

"""

input_dimension = train_images.shape[1:] # Input data shape: (width_px,height_px,#Channels)
encoding_dimension = 128        # just the number of neurons in the deep layers

# --------------------------    1. Encoder    ---------------------------------

encoder = tf.keras.Sequential()
encoder.add(Flatten(input_shape = input_dimension))
encoder.add(Dense(encoding_dimension,activation='relu'))
encoder.add(Dense(encoding_dimension//2, activation='relu'))

# --------------------------    2. Decoder    ---------------------------------

decoder = tf.keras.Sequential()
decoder.add(Dense(encoding_dimension//2,activation='relu'))
decoder.add(Dense(encoding_dimension,activation='relu'))
decoder.add(Dense(int(np.prod(input_dimension)),activation='sigmoid'))
decoder.add(Reshape(input_dimension))

# --------------------------    3. Autoencoder    ---------------------------------

auto_encoder = tf.keras.Sequential()
auto_encoder.add(encoder)
auto_encoder.add(decoder)

# -------------------------    4. Compile & Train    ---------------------------------

auto_encoder.compile(optimizer='adam',loss='mean_squared_error')

auto_encoder.fit(train_images,train_images, #train_images is data and label at the same time
                 epochs=1,
                 batch_size=128,
                 shuffle=True,      # shuffling before each epoch
                 validation_data=(test_images,test_images)) # data on which to evaluate metrics 
                                                            # after each epoch (i.e. loss)


# -----------------------------------------------------------------------------
#                                 RECONSTRUCTION
# -----------------------------------------------------------------------------

image = test_images[9]
image_reshape = image.reshape(1,*input_dimension)
reconstructed_image = auto_encoder.predict(image_reshape)

# -----------------------------------------------------------------------------
#                                 VISUALIZATION
# -----------------------------------------------------------------------------

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(reconstructed_image[0],cmap='gray')
plt.title('Reconstruction')
plt.axis('off')
plt.show()

















































