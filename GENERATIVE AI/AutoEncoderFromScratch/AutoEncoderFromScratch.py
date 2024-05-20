#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:11:01 2024

@author: cornelius
"""

# -----------------------------------------------------------------------------
#                                 IMPORTS
# -----------------------------------------------------------------------------

from DeepNNClass import NeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
#                             DATA PREPARATION
# -----------------------------------------------------------------------------

mnist = pd.read_csv('MNIST_8x8px_images')

features = mnist.drop(columns=['labels']).to_numpy()
features = features.astype('float32')/255.0

labels = mnist['labels'].to_numpy()


# Take only the images that represent a chosen number
features = [features[i] for i in range(len(features)) if labels[i]==9]
# Take the last image as a test image to check how the auto encoder reconstructs
# unknown data it has not been trained on
test_image = features[-1]
# Features are now all images apart from the last one
features = features[:-1]




# -----------------------------------------------------------------------------
#                          SETTING UP PARAMETERS
# -----------------------------------------------------------------------------

input_nodes = len(features[0])
output_nodes = input_nodes
learning_rate = 1e-2
epochs = 100

# -----------------------------------------------------------------------------
#                               AUTO ENCODER
# -----------------------------------------------------------------------------

n = NeuralNetwork(input_nodes, learning_rate)
n.layer(32)
n.layer(16)
n.layer(8)      # Bottle Neck Layer
n.layer(8)      # Bottle Neck Layer
n.layer(16)
n.layer(32)
n.layer(input_nodes)




n.fit_autoencoder(features, epochs)
pred  = n.predict_autoencoder(features[-1])



# -----------------------------------------------------------------------------
#                               VISUALIZATION
# -----------------------------------------------------------------------------


test_image = np.reshape(test_image, [8,8])
pred = np.reshape(pred,[8,8])



plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(test_image,cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(pred,cmap='gray')
plt.title('Reconstruction')
plt.axis('off')
plt.show()










