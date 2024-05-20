#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:43:02 2024

@author: cornelius
"""

"""
                I wrote 2 classes for constructing feed forward
                neural networks
                The first class (NNClass) can only be used to 
                construct a 3-Layer Network (1 hidden layer)
                The second class (DeepNNClass) can be used to 
                construct a network with multiple layers
                
                In this code I test both networks on the
                8x8 px MNIST Data from sklearn


"""


# -----------------------------------------------------------------------------
#                               IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import NNClass
import DeepNNClass
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# -----------------------------------------------------------------------------
#                           DATA PREPARATION
# -----------------------------------------------------------------------------

mnist = pd.read_csv('MNIST_8x8px_images')



features = mnist.drop(columns=['labels']).to_numpy()
labels = mnist['labels'].to_numpy()

feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.3, random_state=10)

# -----------------------------------------------------------------------------
#                           NEURAL NETWORK ANALYSIS
# -----------------------------------------------------------------------------

input_nodes = len(features[0])
hidden_nodes = 100
output_nodes = len(set(labels))
learning_rate = 1e-3
epochs = 10


print()
# -----------------------------------------------------------------------------
#                           3 LAYER NEURAL NETWORK
# -----------------------------------------------------------------------------

n = NNClass.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.fit(feature_train,label_train,epochs)
prediction = n.predict(feature_test)


# Evaluierung der Vorhersagen
accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction, zero_division=1,average='weighted')
sensitivity = metrics.recall_score(label_test, prediction, zero_division=1,average='weighted')


# Ausgabe der Ergebnisse
print("The accuracy is: ", accuracy)
print("The precison is: ", precision)
print("The recall is: ", sensitivity)

print()

# -----------------------------------------------------------------------------
#                            DEEP NEURAL NETWORK
# -----------------------------------------------------------------------------

nn = DeepNNClass.NeuralNetwork(input_nodes, learning_rate)
nn.layer(100)
nn.layer(100)
nn.layer(1000)
nn.layer(50)
nn.layer(output_nodes)



nn.fit(feature_train,label_train,epochs)

prediction = nn.predict(feature_test)

# Evaluierung der Vorhersagen
accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction, zero_division=1,average='weighted')
sensitivity = metrics.recall_score(label_test, prediction, zero_division=1,average='weighted')

# Ausgabe der Ergebnisse
print("The accuracy is: ", accuracy)
print("The precison is: ", precision)
print("The recall is: ", sensitivity)



# -----------------------------------------------------------------------------

image = feature_test[15]
image = np.reshape(image,[8,8])
plt.imshow(image,cmap='gray')
print("Actual Label",label_test[15])
print("Predicted Label: ",nn.predict([feature_test[15]]))




















